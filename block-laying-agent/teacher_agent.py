import copy

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class TeacherAgent(object):
    def __init__(self, target_tensor):
        self.target_tensor = target_tensor
        pass
    
    def check_block(self, latest_block, grid_tensor):
        """
        Checks if an action's block conflicts with existing blocks on the grid or is out of bounds.
        """
        # Get grid cells to check
        check_cells = latest_block['occupied_cells']
        n_cells, _ = check_cells.shape

        # Loop through all cells that would be occupied by action's block
        for i in range(n_cells):
            # x,y,z grid position to check
            x,y,z = list(check_cells[i])
            
            # x,y,z grid position to check
            x,y,z = list(check_cells[i])

            # Check if block cell is out of bounds
            if (x>=self.target_tensor.shape[0]) or (y>=self.target_tensor.shape[1]) or (z>=self.target_tensor.shape[2]):
                # print(f'ENV: Block Out of Bounds at {x,y,z}!')
                return False
            if (x<0) or (y<0) or (z<0):
                # print(f'ENV: Block Out of Bounds at {x,y,z}!')
                return False
            
            # Check if block cell is already occupied in the grid by another block
            if grid_tensor[x,y,z] == 1:
                # print(f'ENV: Block Conflict at {x,y,z}!')
                return False
        
        # All checks pass for all grid cells of the action's block
        # print(f'ENV: No Block Conflict!')
        return True
    
    def step(self, observation, mask=None):
        """
        TeacherAgen adds a block at a random but valid location (i.e., no conflicts, 
        improves reward because block helps complete target voxel model)
        """
        valid_action = False

        # Select a random position from remaining spaces in target voxel model not yet filled
        # (intersection of target voxel model filled cells and unfilled grid cells)
        # Create a mask tensor to identify cells containing the value 1
        mask1 = (self.target_tensor == 1)

        # Find indices corresponding to where a target voxel grid cell is filled 
        indices1 = torch.nonzero(mask1)

        grid_tensor = observation['grid_tensor']

        # Create a mask tensor identifying unfilled grid cells among target voxel grid cells
        mask2 =  (grid_tensor[indices1[:, 0], indices1[:, 1], indices1[:, 2]] == 0)

        # Find indices corresponding to intersection of target voxel model filled cells and unfilled grid cells
        indices2 = torch.nonzero(mask2)

        # Record untried block types and orientations
        untried_block_orients = {
            "2x1": [0,1],
            "3x1": [0,1],
            "4x1": [0,1],
            "2x2": [0,1],
            "3x2": [0,1],
            "4x2": [0,1]
        }

        # Record untried cells and blocks
        untried_cells_blocks = {}
        for index in indices2[:,0]:
            # copy untried_block_orients to each cell 
            # that's part of the target voxel model but currently unfilled
            untried_cells_blocks[index.item()] = copy.deepcopy(untried_block_orients)

        
        while (not valid_action):

            # When it is difficult to fill the remaining cells,
            # If all blocks and cells have been tried
            # then the episode should just terminate
            if len(list(untried_cells_blocks.keys())) == 0:
                # Environment doesn't add a block so return None
                return None
            
            indices3 = list(untried_cells_blocks.keys())
            # Randomly select one index from the list of indices of target voxel model cells not yet filled
            selected_cell = indices3[np.random.randint(0,len(indices3))]
            selected_location = indices1[selected_cell]

            # Select a random block type
            block_type_i = np.random.randint(0, len(list(untried_cells_blocks[selected_cell].keys())))
            block_type = list(untried_cells_blocks[selected_cell].keys())[block_type_i]

            # Select a random orientation
            orientation_i = np.random.randint(0, len(untried_cells_blocks[selected_cell][block_type]))
            orientation = untried_cells_blocks[selected_cell][block_type][orientation_i]

            # Get x y z position of the grid cell at the random index
            grid_x, grid_y, grid_z = selected_location[0].item(), selected_location[1].item(), selected_location[2].item()

            # Based on random block type, orientation, and position
            # determine occupied cells of block
            if orientation == 0:
                grid_position = np.array([grid_x,grid_y,grid_z])
                occupied_cells = grid_position + BLOCK_DEFINITIONS[block_type]['o0_cells']
            if orientation == 1:
                grid_position = np.array([grid_x,grid_y,grid_z])
                occupied_cells = grid_position + BLOCK_DEFINITIONS[block_type]['o1_cells']

            # Create a dictionary defining the random block
            latest_block = {
                "block_type": block_type,
                "block_type_i": block_type_i,
                "grid_position": grid_position,
                "orientation": orientation,
                "occupied_cells": occupied_cells,
            }

            # Check if valid block added by environment does not conflict with existing blocks
            if (self.check_block(latest_block, grid_tensor)):
                valid_action = True
            else:
                # Record all intersection cells that have been tried
                # Record all block types and orientations that have been tried
                untried_cells_blocks[selected_cell][block_type].pop(orientation_i)
                if len(untried_cells_blocks[selected_cell][block_type]) == 0:
                    untried_cells_blocks[selected_cell].pop(block_type)
                if len(list(untried_cells_blocks[selected_cell].keys())) == 0:
                    untried_cells_blocks.pop(selected_cell)

        # return action to the environment for its next step
        return (block_type_i, orientation, grid_x, grid_y, grid_z)
    
    
