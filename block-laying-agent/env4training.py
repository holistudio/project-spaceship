import numpy as np
import torch
import binvox_rw
import datetime

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

ShapeNetIDs = ['02843684']
# vox_files = ['model_normalized.solid.binvox']
vox_files = ['model_normalized.surface.binvox']

class BlockTrainingEnvironment(object):
    def __init__(self):
        super().__init__()
        self.grid_sizes=GRID_SIZES
        self.num_orient=NUM_ORIENTATION
        self.block_info_size=BLOCK_INFO
        self.block_types=BLOCK_TYPES

        # ShapeNetID as integer
        self.ShapeNetID = 0
        self.vox_file = ''
        
        # Load voxel model
        self.target_vox_tensor, self.sum_filled, self.sum_unfilled = torch.zeros((NUM_X,NUM_Y,NUM_Z), dtype=torch.long, device=device), 0, 0

        # Initialize state
        self.state = torch.ones((BLOCK_INFO,NUM_X,NUM_Y,NUM_Z), dtype=torch.long, device=device) * -1
        # state = torch.randint(low=-1, high=40, size=(BLOCK_INFO,NUM_X,NUM_Y,NUM_Z), dtype=torch.long)
        self.state[0,:,:,:] = self.ShapeNetID

        self.grid_tensor = torch.zeros(self.grid_sizes, dtype=torch.long, device=device) # just tracks which cells are occupied

        self.block_seq_index = 0

        self.reward = 0
        self.correct_score = 1
        self.blank_score = 0.1*self.correct_score
        self.incorrect_penalty = self.correct_score

        self.perc_complete = 0
        self.terminal = False

        self.log = {
            "latest_block": {},
            "block_conflict": False,
        }

    def load_vox_model(self, vox_file):
        with open(vox_file, 'rb') as f:
            print('=LOADING VOXEL MODEL=')
            exp_vox = binvox_rw.read_as_3d_array(f) # Expected voxels object

        # Scale down the voxel model and generate a target voxel tensor
        scale_down_factor = 2
        max_x, max_y, max_z = exp_vox.data.shape

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

        fill_mask = (target_vox_tensor == 1)
        sum_filled = torch.sum(fill_mask).item()

        unfill_mask = (target_vox_tensor == 0)
        sum_unfilled = torch.sum(unfill_mask).item()

        # print(NUM_X*NUM_Y*NUM_Z)
        print(f'Total Filled Cells = {sum_filled}')
        # print(sum_unfilled)
        print(f'Model Filled Percent = {sum_filled*100/(NUM_X*NUM_Y*NUM_Z):.2f}%')
        return target_vox_tensor, sum_filled, sum_unfilled

    def reset(self):
        self.__init__()
        select_ID = np.random.randint(0,len(ShapeNetIDs))
        # ShapeNetID as integer
        self.ShapeNetID = int(ShapeNetIDs[select_ID])
        self.vox_file = vox_files[select_ID]
        
        # Load voxel model
        self.target_vox_tensor, self.sum_filled, self.sum_unfilled = self.load_vox_model(self.vox_file)

        # Initialize state
        self.state = torch.ones((BLOCK_INFO,NUM_X,NUM_Y,NUM_Z), dtype=torch.long, device=device) * -1
        # state = torch.randint(low=-1, high=40, size=(BLOCK_INFO,NUM_X,NUM_Y,NUM_Z), dtype=torch.long)
        self.state[0,:,:,:] = self.ShapeNetID

        # Account for target tensor in state as "extra-important missing cells"
        self.state[2:5,:,:,:] = self.state[2:5,:,:,:] - 9*self.target_vox_tensor # x,y,z values only

        self.correct_score = 10/self.sum_filled
        self.blank_score = 0.01*self.correct_score
        self.incorrect_penalty = 10*self.correct_score
        self.perc_complete = 0
        self.terminal = False
        max_reward = (self.correct_score * self.sum_filled) + self.blank_score * self.sum_unfilled
        print(f'Max Reward = {max_reward:.2f}')
        print()

        print('=PLACING BLOCKS=')

        self.log = {
            "latest_block": {
                "block_type": "None",
                "x": -1,
                "y": -1,
                "z": -1,
                "orientation": -1,
            },
            "block_conflict": False,
        }
        return self.state, self.reward, self.terminal

    def no_block_conflict(self, actions):
        check_cells = actions['occupied_cells']
        n_cells, _ = check_cells.shape
        for i in range(n_cells):
            x,y,z = list(check_cells[i])
            if (x>=self.grid_sizes[0]) or (y>=self.grid_sizes[1]) or (z>=self.grid_sizes[2]):
                # print(f'! Block Out of Bounds at {x,y,z} !')
                return False
            if (x<0) or (y<0) or (z<0):
                # print(f'! Block Out of Bounds at {x,y,z} !')
                return False
            if self.grid_tensor[x,y,z] == 1:
                # print(f'! Block Conflict at {x,y,z} !')
                return False
        return True

    def add_block(self, actions):
        pos_x, pos_y, pos_z = actions["grid_position"]
        occupied_cells = actions['occupied_cells']
        n_cells, _ = occupied_cells.shape
        for i in range(n_cells):
            x,y,z = list(occupied_cells[i])
            self.state[:,x,y,z] = torch.tensor([self.ShapeNetID, actions["block_type_i"], pos_x, pos_y, pos_z,
                                                actions['orientation'], self.block_seq_index], dtype=torch.long, device=device)
            self.grid_tensor[x,y,z] = 1
        return self.state

    def calc_reward(self, diff_tensor):
        rew = 0
        
        # value of 0 means either true positive (block where should be a block) or true negative (blank where should be blank)
        zero_indices = torch.nonzero(diff_tensor == 0)
        target_values = self.target_vox_tensor[zero_indices[:, 0], zero_indices[:, 1], zero_indices[:, 2]]
        fill_mask = (target_values == 1)
        n_fill = torch.sum(fill_mask).item()
        n_empty = len(zero_indices) - n_fill
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

        perc_complete = n_fill/self.sum_filled
        return rew, perc_complete

    def determine_terminal(self, diff_tensor):
        if torch.all(diff_tensor == 0):
            return True
        # if self.block_seq_index > 100:
        if self.block_seq_index > self.sum_filled:
            print('! Number of moves exceeded !')
            return True
        return False

    def step(self, agent_env_actions):
        """
        Parameter:
        agent_env_actions - Agent's actions for placing a block specified in a tuple easy for the BlockEnvironment to easily interpret

        Returns: 
        next_state - next state of BlockEnvironment
        reward - Calculated reward
        terminal - if episode should terminate or not
        """
        block_type_i, orientation, grid_x, grid_y, grid_z = agent_env_actions
        block_type = list(BLOCK_DEFINITIONS.keys())[block_type_i]

        if orientation == 0:
            grid_position = np.array([grid_x,grid_y,grid_z])
            occupied_cells = grid_position + BLOCK_DEFINITIONS[block_type]['o0_cells']
        if orientation == 1:
            grid_position = np.array([grid_x,grid_y,grid_z])
            occupied_cells = grid_position + BLOCK_DEFINITIONS[block_type]['o1_cells']

        actions = {
            "block_type": block_type,
            "block_type_i": block_type_i,
            "grid_position": grid_position,
            "orientation": orientation,
            "occupied_cells": occupied_cells
        }

        self.log["latest_block"] = {
            "block_type": block_type,
            "x": int(grid_x),
            "y": int(grid_y),
            "z": int(grid_z),
            "orientation": int(orientation),
        }
    

        if (self.no_block_conflict(actions)):
            # print(f'Agent places {block_type} block at {grid_position}, orientation={orientation}')
            next_state = self.add_block(actions)
        else:
            next_state = self.state

            diff_tensor = self.target_vox_tensor - self.grid_tensor

            block_conflict_penalty = -1000*self.incorrect_penalty

            self.log["block_conflict"] = True

            if self.block_seq_index % 50 == 0:
                print(f'{datetime.datetime.now()}, Block {self.block_seq_index}, Reward = {self.reward:.2f}, Percent complete = {self.perc_complete*100:.2f}%')

            self.block_seq_index += 1
            
            self.terminal = self.determine_terminal(diff_tensor)
            
            return next_state, block_conflict_penalty, self.terminal
        
        diff_tensor = self.target_vox_tensor - self.grid_tensor

        self.reward, self.perc_complete = self.calc_reward(diff_tensor)
        
        self.log["block_conflict"] = False
        
        # print(f'Reward = {reward}')
        if self.block_seq_index % 50 == 0:
            print(f'{datetime.datetime.now()}, Block {self.block_seq_index}, Reward = {self.reward:.2f}, Percent complete = {self.perc_complete*100:.2f}%')

        self.block_seq_index += 1

        self.terminal = self.determine_terminal(diff_tensor)

        return next_state, self.reward, self.terminal