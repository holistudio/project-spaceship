import numpy as np
import torch
import binvox_rw
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

NUM_X = 64
NUM_Y = 64
NUM_Z = 64
GRID_SIZES = (NUM_X, NUM_Y, NUM_Z)

NUM_ORIENTATION = 2

BLOCK_INFO = 7

BLOCK_DEFINITIONS = {
    "2x1":{
        "length":2,
        "width":1,
        "height":1,
        "o0_cells": np.array([[0,0,0],[1,0,0]]),
        "o1_cells": np.array([[0,0,0],[0,0,1]])
    },
    "3x1":{
        "length":3,
        "width":1,
        "height":1,
        "o0_cells": np.array([[0,0,0],[1,0,0],[2,0,0]]),
        "o1_cells": np.array([[0,0,0],[0,0,1],[0,0,2]])
    },
    "4x1":{
        "length":4,
        "width":1,
        "height":1,
        "o0_cells": np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]]),
        "o1_cells": np.array([[0,0,0],[0,0,1],[0,0,2],[0,0,3]])
    },
    "2x2":{
        "length":2,
        "width":2,
        "height":1,
        "o0_cells": np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]]),
        "o1_cells": np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]])
    },
    "3x2":{
        "length":2,
        "width":2,
        "height":1,
        "o0_cells": np.array([[0,0,0],[1,0,0],[2,0,0],[2,0,1],[1,0,1],[0,0,1]]),
        "o1_cells": np.array([[0,0,0],[1,0,0],[1,0,1],[1,0,2],[0,0,2],[0,0,1]])
    },
    "4x2":{
        "length":2,
        "width":2,
        "height":1,
        "o0_cells": np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[3,0,1],[2,0,1],[1,0,1],[0,0,1]]),
        "o1_cells": np.array([[0,0,0],[1,0,0],[1,0,1],[1,0,2],[1,0,3],[0,0,3],[0,0,2],[0,0,1]])
    }
}
BLOCK_TYPES = len(BLOCK_DEFINITIONS.keys())

ShapeNetIDs = ['04099429']
# vox_files = ['model_normalized.solid.binvox']
vox_files = ['model_normalized.surface.binvox']

def get_random_folder(base_folder):
    # List all entries in the base folder
    entries = os.listdir(base_folder)
    
    # Filter out entries that are not directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join(base_folder, entry))]
    
    # Select a random directory
    if directories:
        return random.choice(directories)
    else:
        return None
    
class BlockTrainingEnvironment(object):
    def __init__(self, reset=False):
        super().__init__()
        self.grid_sizes=GRID_SIZES
        self.num_orient=NUM_ORIENTATION
        self.block_info_size=BLOCK_INFO
        self.block_types=BLOCK_TYPES

        # ShapeNetID as integer
        self.ShapeNetID = 0
        self.vox_file = ''
        
        # Initialize state with -1s representing blank cells
        self.state = torch.ones((BLOCK_INFO,NUM_X,NUM_Y,NUM_Z), dtype=torch.long, device=device) * -1

        # Initialize tensor for tracking which grid cells are occupied
        self.grid_tensor = torch.zeros(self.grid_sizes, dtype=torch.long, device=device) 

        # Initialize index tracking how many blocks have been attempted by Agent
        self.block_seq_index = 0

        # Initialize reward
        self.reward = 0

        # Track percent complete and terminal
        self.perc_complete = 0
        self.terminal = False

        # Initialize log
        self.log = {
            "latest_agent_block": {
                "block_type": "None",
                "x": -1,
                "y": -1,
                "z": -1,
                "orientation": -1,
                "block_conflict": False,
            },
            "latest_env_block": {
                "block_type": "None",
                "x": -1,
                "y": -1,
                "z": -1,
                "orientation": -1,
                "block_conflict": False,
            },
        }

        if reset:
            # Randomly select a ShapeNetID
            select_ID = np.random.randint(0,len(ShapeNetIDs))

            # ShapeNetID as integer
            self.ShapeNetID = int(ShapeNetIDs[select_ID])
            
            # Select corresponding voxel model filepath
            shape_dir = os.path.join('target_models',str(ShapeNetIDs[select_ID]))
            model_dir = os.path.join(shape_dir,get_random_folder(shape_dir))
            self.vox_file = os.path.join(model_dir,'models','model_normalized.surface.binvox')
            
            # Load voxel model
            self.target_vox_tensor, self.sum_filled, self.sum_unfilled = self.load_vox_model(self.vox_file)
            
            # Take difference between target voxel grid cells and current grid cells occupied
            self.diff_tensor = self.target_vox_tensor - self.grid_tensor

            # First dimension across the entire state tensor set to ShapeNetID
            self.state[0,:,:,:] = self.ShapeNetID
            
            # Account for target tensor in state as "extra-important missing cells"
            self.state[2:5,:,:,:] = self.state[2:5,:,:,:] - 9*self.target_vox_tensor # x,y,z values only

            # Reward/penalty system
            self.correct_score = 10/self.sum_filled
            self.blank_score = 0.01*self.correct_score
            self.incorrect_penalty = 10*self.correct_score

            # Calculate max possible reward based on the target voxel model
            max_reward = (self.correct_score * self.sum_filled) + self.blank_score * self.sum_unfilled
            print(f'Max Reward = {max_reward:.2f}')
            print()

    def load_vox_model(self, vox_file):
        """
        Parameter:
        vox_file - Filepath for the voxel model from ShapeNet

        Returns: 
        target_vox_tensor - Tensor of target voxel model, only specifying whether a grid cell is filled/unfilled
        sum_filled - Total number of cells in the grid filled in by the target voxel model
        sum_unfilled - Total number of cellsin the grid unfilled in by the target voxel model
        """

        # Load voxel model using binvox_rw library
        with open(vox_file, 'rb') as f:
            print('== LOADING VOXEL MODEL ==')
            exp_vox = binvox_rw.read_as_3d_array(f) # Expected voxels object

        # Scale down the voxel model and generate a target voxel tensor
        scale_down_factor = 2
        max_x, max_y, max_z = exp_vox.data.shape

        # Initialize target voxel tensor based on the voxel model and scale_down_factor
        s = scale_down_factor
        x = 0
        vx = 0

        target_vox_tensor = torch.zeros((int(max_x/s),int(max_y/s),int(max_z/s)), dtype=torch.long, device=device)

        while (x<max_x):
            y = 0
            vy = 0
            while (y<max_y):
                z = 0
                vz = 0
                while (z<max_z):
                    boolean_list = exp_vox.data[x:x+s,y:y+s,z:z+s].ravel().tolist()
                    count_true = sum(boolean_list)
                    if(count_true >= 3):
                        target_vox_tensor[vx,vy,vz] = 1
                    z += scale_down_factor
                    vz +=1
                y += scale_down_factor
                vy += 1
            x += scale_down_factor
            vx += 1

        # Calculate how many grid cells are filled/unfilled by voxel model
        fill_mask = (target_vox_tensor == 1)
        sum_filled = torch.sum(fill_mask).item()

        unfill_mask = (target_vox_tensor == 0)
        sum_unfilled = torch.sum(unfill_mask).item()

        # Output target model voxel properties
        # print(NUM_X*NUM_Y*NUM_Z)
        print(f'Total Filled Cells = {sum_filled}')
        # print(sum_unfilled)
        print(f'Model Filled Percent = {sum_filled*100/(NUM_X*NUM_Y*NUM_Z):.2f}%')
        return target_vox_tensor, sum_filled, sum_unfilled

    def reset(self):
        """
        Reset BlockTrainingEnvironment with new target voxel model from ShapeNet
        Returns: state, reward, terminal
        """

        # Re-initialize
        self.__init__(reset=True)

        return self.state, self.reward, self.terminal

    def no_block_conflict(self, actions):
        """
        Checks if an action's block conflicts with existing blocks on the grid or is out of bounds.

        Parameters:
        actions - dictionary containing the occupied_cells of the action's block

        Returns:
        True or False depending on if other grid cells are the same as occupied cells.
        """

        # Get grid cells to check
        check_cells = actions['occupied_cells']
        n_cells, _ = check_cells.shape

        # Loop through all cells that would be occupied by action's block
        for i in range(n_cells):
            # x,y,z grid position to check
            x,y,z = list(check_cells[i])

            # Check if block cell is out of bounds
            if (x>=self.grid_sizes[0]) or (y>=self.grid_sizes[1]) or (z>=self.grid_sizes[2]):
                # print(f'! Block Out of Bounds at {x,y,z} !')
                return False
            if (x<0) or (y<0) or (z<0):
                # print(f'! Block Out of Bounds at {x,y,z} !')
                return False
            
            # Check if block cell is already occupied in the grid by another block
            if self.grid_tensor[x,y,z] == 1:
                # print(f'! Block Conflict at {x,y,z} !')
                return False
        
        # All checks pass for all grid cells of the action's block
        return True

    def add_block(self, actions):
        """
        Adds action's block to the grid and updates the state tensor.
        Updates grid_tensor and diff_tensor accordingly

        Parameters:
        actions - dictionary containing attributes of the block

        Returns:
        state - tensor used by agent for selecting the next action/block
        """

        # Get grid position of the block
        pos_x, pos_y, pos_z = actions["grid_position"]

        # Get occupied cells of the block
        occupied_cells = actions['occupied_cells']
        n_cells, _ = occupied_cells.shape

        # Loop through occupied cells and update state tensor
        for i in range(n_cells):
            x,y,z = list(occupied_cells[i])
            self.state[:,x,y,z] = torch.tensor([self.ShapeNetID, actions["block_type_i"], pos_x, pos_y, pos_z,
                                                actions['orientation'], self.block_seq_index], dtype=torch.long, device=device)
            self.grid_tensor[x,y,z] = 1
        
        # Take difference between target voxel grid cells and current grid cells occupied
        self.diff_tensor = self.target_vox_tensor - self.grid_tensor
        return self.state
    
    def env_add_block(self):
        """
        Environment adds a block at a random but valid location (i.e., no conflicts, consistent with target voxel model)

        Returns: Updated state for agent with the added block
        """
        # print(f'{datetime.datetime.now()}, Block {self.block_seq_index}, Environment attempts to add another block...')
        valid_action = False

        while (not valid_action):
            # Select a random block type
            block_type_i = np.random.randint(0, BLOCK_TYPES)
            block_type = list(BLOCK_DEFINITIONS.keys())[block_type_i]

            # Select a random orientation
            orientation = np.random.randint(0, 2)

            # Select a random position based on target voxel model's grid cells
            # Create a mask tensor to identify cells containing the value 1
            mask = (self.target_vox_tensor == 1)

            # Find indices corresponding to where a target voxel grid cell is filled 
            indices = torch.nonzero(mask)

            # Randomly select one index from the list of indices of filled target voxel grid cells
            selected_location = indices[torch.randint(0, indices.size(0), (1,))]

            # Get x y z position of the grid cell at the random index
            grid_x, grid_y, grid_z = selected_location[0][0].item(), selected_location[0][1].item(), selected_location[0][2].item()

            # Based on random block type, orientation, and position
            # determine occupied cells of block
            if orientation == 0:
                grid_position = np.array([grid_x,grid_y,grid_z])
                occupied_cells = grid_position + BLOCK_DEFINITIONS[block_type]['o0_cells']
            if orientation == 1:
                grid_position = np.array([grid_x,grid_y,grid_z])
                occupied_cells = grid_position + BLOCK_DEFINITIONS[block_type]['o1_cells']

            # Check if random block is valid

            # Create a dictionary defining the random block
            env_actions = {
                "block_type": block_type,
                "block_type_i": block_type_i,
                "grid_position": grid_position,
                "orientation": orientation,
                "occupied_cells": occupied_cells,
                "author": "environment"
            }

            # Calculate the reward before this block is placed
            # for later comparisons
            grid_tensor_copy = self.grid_tensor.clone().cpu()
            target_vox_tensor_copy = self.target_vox_tensor.clone().cpu()
            diff_tensor_before = target_vox_tensor_copy - grid_tensor_copy
            reward_before, _ = self.calc_reward(diff_tensor_before)

            # Check if valid block added by environment does not conflict with existing blocks
            if (self.no_block_conflict(env_actions)):
                # Loop through the occupied cells of the random block
                n_cells, _ = occupied_cells.shape
                for i in range(n_cells):
                    x,y,z = list(occupied_cells[i])
                    grid_tensor_copy[x,y,z] = 1
                
                # Calculate the reward that results from this random block's placement
                diff_tensor_after = target_vox_tensor_copy - grid_tensor_copy
                reward_after, _ = self.calc_reward(diff_tensor_after)

                # Check if valid block results in an increase in reward
                if reward_after > reward_before:
                    # Only break out of the while loop
                    # if block has no conflicts and increases the reward
                    valid_action = True
        
        # Environment adds a new block in random valid position
        # Log latest block by environment
        self.log["latest_env_block"] = {
            "block_type": block_type,
            "x": int(grid_x),
            "y": int(grid_y),
            "z": int(grid_z),
            "orientation": int(orientation),
            "block_conflict": False
        }

        return self.add_block(env_actions)

    def calc_reward(self, diff_tensor):
        """
        Calculates the reward based on the difference between the target voxel model's grid cells and current grid cells occupied

        Parameters:
        diff_tensor - tensor of same shape as the 3D grid, containing 0s, 1s, and -1s signifying matches with target voxel model

        Returns:
        rew - agent reward calculated by environment
        perc_complete - Percent completion of the model based on number of correctly filled cells / total number of filled cells in target voxel model
        """
        rew = 0
        
        # value of 0 means either true positive (block where should be a block) or true negative (blank where should be blank)
        zero_indices = torch.nonzero(diff_tensor == 0)

        # Filter the target voxel model tensor to focus only on cells where current grid matches target voxel model grid cells
        target_values = self.target_vox_tensor[zero_indices[:, 0], zero_indices[:, 1], zero_indices[:, 2]]

        # Mask identifying grid cells are supposed to be filled (not blank) based on target voxel model
        fill_mask = (target_values == 1)

        # Tally up correctly filled cells
        n_fill = torch.sum(fill_mask).item()

        # Tally up correct unfilled/blank cells
        n_empty = len(zero_indices) - n_fill

        # Assign rewards based on correctly filled, correctly blank cells
        rew += n_fill * self.correct_score
        rew += n_empty * self.blank_score

        # value of -1 means false positive (block placed by agent but should be blank)
        neg_1_mask = (diff_tensor == -1)
        n_fp = torch.sum(neg_1_mask).item()
        rew -= n_fp * self.incorrect_penalty

        # value of +1 means false negative (should block but none placed by agent)
        pos_1_mask = (diff_tensor == 1)
        n_fn = torch.sum(pos_1_mask).item()
        rew -= n_fn * self.incorrect_penalty

        # Calculate percent complete
        perc_complete = n_fill/self.sum_filled
        return rew, perc_complete

    def determine_terminal(self, diff_tensor, perc_complete):
        """
        Check if episode should terminate, either because the blocks complete the model or the number of attempts have exceeded a limit.

        Parameters:
        diff_tensor - tensor of same shape as the 3D grid, containing 0s, 1s, and -1s signifying matches with target voxel model

        Returns:
        True or False - depending on whether blocks complete the model or the number of attempts have exceeded a limit.
        """

        # If all filled and unfilled cells match the target model
        if perc_complete >= 1.0:
            return True
        if torch.all(diff_tensor == 0):
            return True
        
        # If number of attempts exceed 100 blocks
        #if self.block_seq_index > 100:

        # If number of attempts exceed the total number of filled cells for the target voxel model
        if self.block_seq_index > self.sum_filled:
            print('! Number of moves exceeded !')
            return True
        
        # Otherwise, episode continues
        return False

    def step(self, agent_env_actions):
        """
        BlockTrainingEnvironment one step forward.

        Parameters:
        agent_env_actions - Agent's actions for placing a block specified in a tuple easy for the BlockTrainingEnvironment to easily interpret

        Returns: 
        next_state - next state of BlockEnvironment
        reward - Calculated reward
        terminal - if episode should terminate or not
        """

        # Convert agent_env_actions into block type, position and orientation
        block_type_i, orientation, grid_x, grid_y, grid_z = agent_env_actions

        # Get block type as specified in BLOCK_DEFINITIONS dictionary
        block_type = list(BLOCK_DEFINITIONS.keys())[block_type_i]

        # Based on grid position and orientation, determine cells occupied in the grid
        if orientation == 0:
            grid_position = np.array([grid_x,grid_y,grid_z])
            occupied_cells = grid_position + BLOCK_DEFINITIONS[block_type]['o0_cells']
        if orientation == 1:
            grid_position = np.array([grid_x,grid_y,grid_z])
            occupied_cells = grid_position + BLOCK_DEFINITIONS[block_type]['o1_cells']

        # Store agent block actions into a dictionary
        actions = {
            "block_type": block_type,
            "block_type_i": block_type_i,
            "grid_position": grid_position,
            "orientation": orientation,
            "occupied_cells": occupied_cells,
            "author": "agent"
        }

        # Log latest block by agent
        self.log["latest_agent_block"] = {
            "block_type": block_type,
            "x": int(grid_x),
            "y": int(grid_y),
            "z": int(grid_z),
            "orientation": int(orientation),
            "block_conflict": False
        }

        # Check for conflicts between agent's proposed block and existing blocks in BlockTrainingEnvironment
        if (self.no_block_conflict(actions)):
            # print(f'Agent places {block_type} block at {grid_position}, orientation={orientation}')

            # If there are no conflicts, BlockTrainingEnvironment adds agent block to the grid
            next_state = self.add_block(actions)
            self.block_seq_index += 1

            # Environment adds a block in a random valid location
            next_state = self.env_add_block()
            self.block_seq_index += 1

            # Calculate reward based on how well occupied grid cells match target voxel grid cells.
            self.reward, self.perc_complete = self.calc_reward(self.diff_tensor)

            # Log no conflict for agent block
            self.log["latest_agent_block"]["block_conflict"] = False
        else:
            # If there is a conflict with existing blocks
            # Increment block index
            self.block_seq_index += 1

            # Environment state remains unchanged
            next_state = self.state
            # self.grid_tensor, diff_tensor, and perc_complete remain unchanged as well

            # Set reward to the block conflict penalty (should be >> typical +reward or -incorrect_penalties)
            self.reward = -1000*self.incorrect_penalty

            # Log conflict
            self.log["latest_agent_block"]["block_conflict"] = True
            # Environment doesn't add a block so log default values for env_block
            self.log["latest_env_block"] = {
                "block_type": "None",
                "x": -1,
                "y": -1,
                "z": -1,
                "orientation": -1,
                "block_conflict": False,
            }

        # Check if episode terminates
        self.terminal = self.determine_terminal(self.diff_tensor, self.perc_complete)
        
        return next_state, self.reward, self.terminal