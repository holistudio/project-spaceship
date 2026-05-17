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

    def __init__(self, grid_dim=32, max_timesteps=10_000):
        print("INITIALIZING ENVIRONMENT...", end="")

        self.timestep = 0
        self.max_timesteps = max_timesteps
        self.block_id = 0

        self.grid_shape = (grid_dim, grid_dim, grid_dim)
        self.voxel_grid = np.zeros(self.grid_shape)
        self.num_block_types = len(BLOCK_TYPES)
        self.num_orientations = 2
        self.action_space = np.arange(self.num_block_types * 
                                      self.num_orientations * 
                                      self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2])
        self.action_mask, self.edge_invalid_actions = self._initial_action_mask()

        self.players = ['user', 'agent']
        self._player_selector = iter(self.players)
        self.current_player = next(self._player_selector)

        self.observation = {
            "observation": {
                "user_blocks": [],
                "agent_blocks": [],
                "voxel_grid": self.voxel_grid
            },
            "action_mask": self.action_mask
        }
        self.reward = 0
        self.termination = False
        self.truncation = False

        print("done\n\n")
        pass

    def _initial_action_mask(self):
        action_mask = np.ones(self.action_space.shape, dtype=bool)
        invalid_idx = set()
        
        X, Y, Z = self.grid_shape
        xs, zs = list(range(X)), list(range(Z))
        xs = xs[:3] + xs[-3:]
        zs = zs[:3] + zs[-3:]
        for block_type_i, block_type in enumerate(BLOCK_DEFINITIONS):
            for orientation in range(self.num_orientations):
                # only check edge of X and Z ranges [3, -3:]
                for x in xs:
                    # check every Y coordinate (up/down)
                    for y in range(Y):
                        for z in zs:

                            # check if the specific block position + orientation is out-of-bounds
                            position = np.array([x,y,z])
                            if orientation == 0:
                                cells = BLOCK_DEFINITIONS[block_type]["o0_cells"]
                            else:
                                cells = BLOCK_DEFINITIONS[block_type]["o1_cells"]
                            cells = position + cells
                            out_bounds = np.count_nonzero(
                                (cells[:, 0] < 0) | (cells[:, 0] > X) |
                                (cells[:, 1] < 0) | (cells[:, 1] > Y) |
                                (cells[:, 2] < 0) | (cells[:, 2] > Z) 
                            )
                            if out_bounds > 0:
                                # assumes the lexicographic order: block_type → orientation → x → y → z
                                idx = (block_type_i * (self.num_orientations * X * Y * Z) +
                                    orientation * (X * Y * Z) +
                                    x * (Y * Z) +
                                    y * (Z) +
                                    z 
                                )
                                action_mask[idx] = False
                                invalid_idx.add(idx)
        return action_mask, invalid_idx

    def reset(self):
        print("RESETTING ENVIRONMENT...", end="")

        self.timestep = 0
        self.block_id = 0
        self.voxel_grid = np.zeros(self.grid_shape)
        self.action_space = np.arange(self.num_block_types * 
                                      self.num_orientations * 
                                      self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2])
        self.action_mask, self.edge_invalid_actions = self._initial_action_mask()
        
        
        self._player_selector = iter(self.players)
        self.current_player = next(self._player_selector)

        self.observation = {
            "observation": {
                "user_blocks": [],
                "agent_blocks": [],
                "voxel_grid": self.voxel_grid
            },
            "action_mask": self.action_mask
        }
        self.reward = 0
        self.termination = False
        self.truncation = False

        print("done")
        pass

    def agent_iter(self, max_iter=2**63):
        i = 0
        while i < max_iter and self.players:
            yield self.current_player
            i += 1

    def last(self):
        return self.observation, self.reward, self.termination, self.truncation
    
    def _action_to_placement(self, a):
        X, Y, Z = self.grid_shape
        O = self.num_orientations

        # decompose the action index
        # assumes the lexicographic order: block_type → orientation → x → y → z
        remainder, z = divmod(a, Z)
        remainder, y = divmod(remainder, Y)
        remainder, x = divmod(remainder, X)
        x, y, z = int(x), int(y), int(z)

        remainder, orientation = divmod(remainder, O)
        orientation = int(orientation)

        block_type_i= remainder
        block_type = BLOCK_TYPES[block_type_i]

        return {
            "block_type": block_type,
            "orientation": orientation,
            "position": (x, y, z)
        }   
    
    def sample_action(self, mask):
        if mask is not None:
            actions = self.action_space[mask]
        a = actions[np.random.randint(0,len(actions),())]
        print(a)
        placement = self._action_to_placement(a)
        print(f"{placement["block_type"]} Block: {placement["position"], placement["orientation"]}")
        return a
    
    # close():


    # _check_action(action):
        # if the block (position, orientation) overlaps with existing voxel grid
        # stop and raise error

    # _mask_actions():
        # go through action space
        # if the block (position, orientation) overlaps with existing voxel grid,
        # mask it out

    def step(self, actions=None):
        self.timestep += 1
        # new_blocks = actions["new_blocks"]
        
        # for a in new_blocks:
            # _check_action(a)
            # update voxel grid
            # update action mask
            # update observation

        # if current_player == 'user':
            # user_terminal = actions["done"]
            # if user_terminal:
                # return 

            # update agent_blocks based on change_blocks
            # change_blocks = actions["change_blocks"]
            # _check_action(a)
            # update voxel grid
            # update action mask
            # update observation

            # update reward
            # reward += actions["score"]

        # check truncation
        if self.timestep >= self.max_timesteps:
            print(f"{"!"*5} EPISODE TRUNCATED {"!"*5}\n")
            self.truncation = True


        # iterate to next player
        try:
            self.current_player = next(self._player_selector)
        except StopIteration:
            self._player_selector = iter(self.players)
            self.current_player = next(self._player_selector)
        pass