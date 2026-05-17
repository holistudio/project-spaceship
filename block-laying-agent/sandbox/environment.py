import numpy as np

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

BLOCK_TYPES = list(BLOCK_DEFINITIONS.keys())

class Block(object):
    def __init__(self, block_id, block_type_i, x, y, z, orientation):
        
        self.block_id = block_id
        
        self.block_type = BLOCK_TYPES[block_type_i]
        self.x, self.y, self.z = x, y, z
        self.orientation = orientation 
        

class BlockEnvironment(object):

    def __init__(self, grid_dim=32):
        print("INITIALIZING ENVIRONMENT...", end="")
        self.timestep = 0
        self.block_id = 0

        self.grid_shape = (grid_dim, grid_dim, grid_dim)
        self.voxel_grid = np.zeros(self.grid_shape)
        num_block_types = len(BLOCK_TYPES)
        num_orientations = 2
        self.action_space = np.arange(num_block_types * num_orientations * grid_dim * grid_dim * grid_dim)

        self.players = ['user', 'agent']
        self.current_player = 'user'

        

        # observation = {
            # "observation": {
                # "user_blocks": [],
                # "agent_blocks": [],
                # "voxel_grid": voxel_grid
            # }
            # "actions_mask": self._mask_actions()
        #}
        print("done\n\n")
        pass


# reset():

# agent_iter():

# last():
    # return observation, reward, termination, truncation, info

# close():



# _check_action(action):
    # if the block (position, orientation) overlaps with existing voxel grid
    # stop and raise error

# _mask_actions():
    # go through action space
    # if the block (position, orientation) overlaps with existing voxel grid,
    # mask it out

# step(actions):
    # new_blocks = actions["new_blocks"]
    
    # for a in new_blocks:
        # _check_actions(a)
        # update voxel grid
        # update observation

    # if current_player == 'user':
        # user_terminal = actions["done"]
        # if user_terminal:
            # return 

        # update agent_blocks based on change_blocks
        # change_blocks = actions["change_blocks"]
        # _check_actions(a)

        # update reward
        # reward += actions["score"]

    # pass