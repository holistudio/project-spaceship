import os

import binvox_rw

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')

REW_CORRECT_FILL = 1
REW_INCORRECT = -1

NUM_X = 32
NUM_Y = 32
NUM_Z = 32
GRID_SIZES = (NUM_X, NUM_Y, NUM_Z)

NUM_ORIENTATION = 2

BLOCK_INFO = 5
# block_type_i- which of the 6 types of block occupies the grid cell
# pos_x, pos_y, pos_z -position of the block's anchor (not the x,y,z coordinates of the grid cell itself)
# orientation - 0 or 1 representing the two possible block rotations

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

shape_ids = ['04099429'] # TODO: add more ShapeNet IDs here later...
vox_files = ['model_normalized.surface.binvox'] # use the "hollow" voxel models

def get_random_folder(base_folder):
    # List all entries in the base folder
    entries = os.listdir(base_folder)
    
    # Filter out entries that are not directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join(base_folder, entry))]
    
    # Select a random directory
    if directories:
        ix = np.random.randint(0,len(directories))
        return directories[ix]
    else:
        return None

def load_vox_model(vox_file):
    """
    Parameter:
    vox_file - Filepath for the voxel model from ShapeNet

    Returns: 
    target_tensor - Tensor of target voxel model, only specifying whether a grid cell is filled/unfilled
    total_filled - Total number of cells in the grid filled in by the target voxel model
    total_unfilled - Total number of cellsin the grid unfilled in by the target voxel model
    """

    # Load voxel model using binvox_rw library
    with open(vox_file, 'rb') as f:
        print('ENV: Loading Voxel Model')
        exp_vox = binvox_rw.read_as_3d_array(f) # Expected voxels object

    # Scale down the voxel model and generate a target voxel tensor
    scale_down_factor = 4
    max_x, max_y, max_z = exp_vox.data.shape

    # Initialize target voxel tensor based on the voxel model and scale_down_factor
    s = scale_down_factor
    x = 0
    vx = 0

    target_tensor = torch.zeros((int(max_x/s),int(max_y/s),int(max_z/s)), dtype=torch.long, device=device)

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
                    target_tensor[vx,vy,vz] = 1
                z += scale_down_factor
                vz +=1
            y += scale_down_factor
            vy += 1
        x += scale_down_factor
        vx += 1

    # Calculate how many grid cells are filled/unfilled by voxel model
    fill_mask = (target_tensor == 1)
    total_filled = torch.sum(fill_mask).item()

    unfill_mask = (target_tensor == 0)
    total_unfilled = torch.sum(unfill_mask).item()
    return target_tensor, total_filled, total_unfilled

class BlockEnvironment(object):
    def __init__(self, reset=False):
        # Initialize state with -1s representing blank cells
        self.state = torch.ones((BLOCK_INFO,NUM_X,NUM_Y,NUM_Z), dtype=torch.long, device=device) * -1

        # Initialize tensor for tracking which grid cells are occupied
        self.grid_tensor = torch.zeros((NUM_X,NUM_Y,NUM_Z), dtype=torch.long, device=device) 

        # Initialize index tracking how many blocks have been added
        self.block_ix = 0

        # Initialize reward
        self.reward = 0

        # Track terminal and percent complete
        self.terminal = False
        self.perc_complete = 0

        # Initialize environment log
        self.env_log = {
            "latest_block": {
                "player_id": -1,
                "block_type": "None",
                "x": -1,
                "y": -1,
                "z": -1,
                "orientation": -1,
                "author": "agent_0"
            }
        }

        if reset:
            # Randomly select a ShapeNetID
            ix = np.random.randint(0,len(shape_ids))

            # ShapeNetID as integer
            self.shape_id = int(shape_ids[ix])
            # Select corresponding voxel model filepath
            shape_dir = os.path.join('target_models',str(shape_ids[ix]))
            model_dir = os.path.join(shape_dir,get_random_folder(shape_dir))
            self.vox_file = os.path.join(model_dir,'models','model_normalized.surface.binvox')
            
            self.target_tensor, self.total_filled, self.total_unfilled = load_vox_model(self.vox_file)
            self.diff_tensor = self.target_tensor - self.grid_tensor
            print(f'ENV: Total Filled Cells={self.total_filled}')
            print(f'ENV: Model Filled Percent={self.total_filled*100/(NUM_X*NUM_Y*NUM_Z):.2f}%')
        pass

    def reset(self):
        self.__init__(reset=True)
        pass

    def check_block(self, latest_block):
        """
        Checks if an action's block conflicts with existing blocks on the grid or is out of bounds.

        Parameters:
        latest_block - dictionary containing the occupied_cells of the action's block

        Returns:
        True or False depending on if other grid cells are the same as occupied cells.
        """
        # Get grid cells to check
        check_cells = latest_block['occupied_cells']
        n_cells, _ = check_cells.shape

        # Loop through all cells that would be occupied by action's block
        for i in range(n_cells):
            # x,y,z grid position to check
            x,y,z = list(check_cells[i])

            # Check if block cell is out of bounds
            if (x>=self.grid_tensor.shape[0]) or (y>=self.grid_tensor.shape[1]) or (z>=self.grid_tensor.shape[2]):
                # print(f'ENV: Block Out of Bounds at {x,y,z}!')
                return False
            if (x<0) or (y<0) or (z<0):
                # print(f'ENV: Block Out of Bounds at {x,y,z}!')
                return False
            
            # Check if block cell is already occupied in the grid by another block
            if self.grid_tensor[x,y,z] == 1:
                # print(f'ENV: Block Conflict at {x,y,z}!')
                return False
        
        # All checks pass for all grid cells of the action's block
        # print(f'ENV: No Block Conflict!')
        return True
    
    def add_block(self, latest_block):
        """
        Adds latest_block's block to the grid and updates the state tensor.
        Updates grid_tensor and diff_tensor accordingly

        Parameters:
        latest_block - dictionary containing attributes of the latest block
        """
        # Get grid position of the block
        pos_x, pos_y, pos_z = latest_block["grid_position"]

        # Get occupied cells of the block
        occupied_cells = latest_block['occupied_cells']
        n_cells, _ = occupied_cells.shape

        # Loop through occupied cells and update state tensor
        for i in range(n_cells):
            x,y,z = list(occupied_cells[i])
            self.state[:,x,y,z] = torch.tensor([latest_block["block_type_i"], 
                                                pos_x, pos_y, pos_z,
                                                latest_block['orientation']], 
                                                dtype=torch.long, device=device)
            self.grid_tensor[x,y,z] = 1
        
        # Take difference between target voxel grid cells and current grid cells occupied
        self.diff_tensor = self.target_tensor - self.grid_tensor
        pass

    def score(self):
        """
        Calculates the reward based on the difference between the target voxel model's grid cells and current grid cells occupied
        """
        rew = 0

        # value of 0 means either true positive (block where should be a block) or true negative (blank where should be blank)
        zero_indices = torch.nonzero(self.diff_tensor == 0)

        # Filter the target voxel model tensor to focus only on cells where current grid matches target voxel model grid cells
        target_values = self.target_tensor[zero_indices[:, 0], zero_indices[:, 1], zero_indices[:, 2]]

        # Mask identifying grid cells are supposed to be filled (not blank) based on target voxel model
        fill_mask = (target_values == 1)

        # Tally up correctly filled cells
        n_fill = torch.sum(fill_mask).item()

        # Tally up correct unfilled/blank cells
        n_empty = len(zero_indices) - n_fill

        # Assign rewards based on correctly filled cells
        rew += n_fill * REW_CORRECT_FILL

        # value of -1 means false positive (block placed by agent but should be blank)
        neg_1_mask = (self.diff_tensor == -1)
        n_fp = torch.sum(neg_1_mask).item()
        rew -= n_fp * REW_INCORRECT

        # Calculate percent complete
        self.reward = rew
        self.perc_complete = n_fill/self.total_filled
        pass

    def is_terminal(self):
        """
        Check if episode should terminate, either because the blocks complete the model or the number of attempts have exceeded a limit.
        """
        # If all filled and unfilled cells match the target model
        if self.perc_complete >= 1.0:
            print('ENV: 100% Complete!!')
            return True
        if torch.all(self.diff_tensor == 0):
            print('ENV: 100% Match!!!')
            return True
        
        # If number of attempts exceed the total number of filled cells for the target voxel model
        # then it is likely going to take too long to complete the given voxel model with 100% filled
        if self.block_ix > self.total_filled:
            print('ENV: Number of moves exceeded!')
            return True
        # Otherwise, episode continues
        return False

    def step(self, action):
        """
        BlockTrainingEnvironment one step forward.
        """
        if action is None:
            next_observation = {
                "state": self.state,
                "grid_tensor": self.grid_tensor
            }
            self.terminal = True
            return next_observation
        block_type_i, orientation, grid_x, grid_y, grid_z = action

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
        latest_block = {
            "block_ix": self.block_ix,
            "block_type": block_type,
            "block_type_i": block_type_i,
            "grid_position": grid_position,
            "orientation": orientation,
            "occupied_cells": occupied_cells,
        }
        
        self.env_log["latest_block"] = {
            "block_ix": self.block_ix,
            "block_type": block_type,
            "x": int(grid_x),
            "y": int(grid_y),
            "z": int(grid_z),
            "orientation": int(orientation),
            "author": "agent_0" if self.block_ix % 2 == 0 else "agent_1"
        }

        # Check for conflicts between agent block and existing blocks in BlockEnvironment
        if (self.check_block(latest_block)):
            # If there are no conflicts, BlockTrainingEnvironment adds agent block to the grid
            self.add_block(latest_block)    
            self.score()
        else:
            # If the anget block conflicts with other blocks
            # End the episode 
            self.terminal = True
            print(f'ENV: FAILED! Step {self.block_ix}, Agent places {latest_block["block_type"]} block at {latest_block["grid_position"][0], latest_block["grid_position"][1], latest_block["grid_position"][2]}, orientation={latest_block["orientation"]}')

        next_observation = {
            "state": self.state,
            "grid_tensor": self.grid_tensor
        }
        
        self.block_ix += 1
        return next_observation
    
    def action_mask(self):
        search_range = [1, 2, 3, -1, -2, -3]
        # find all filled cells in grid_tensor

        # add cells to the action mask for all block types and orientations

        # for each filled cell
            # for x+1, +2, +3, -1, -2, -3
                # for y+1, +2, +3, -1, -2, -3
                    # for z+1, +2, +3, -1, -2, -3
                        # initialize unchecked block_types and orientations
                        # check each block type and orientation
                        # if it doesn't work then add it to the action mask
        # return mask
        return None
    
    def last(self):
        observation = {
            "state": self.state,
            "grid_tensor": self.grid_tensor
        }
        info = {}
        info['action_mask'] = self.action_mask()

        if not self.terminal:
            self.terminal = self.is_terminal()
        
        return observation, self.reward, self.terminal, info
    
    