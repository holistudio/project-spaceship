import numpy as np
import torch
import binvox_rw

import CNN_agent as Agent

BLOCK_INFO = 7
NUM_ORIENTATION = 2

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

# ShapeNetID as integer
ShapeNetID = int('02843684')

# Load voxel model
vox_file = '../../datasets/shapenet/02843684/1b73f96cf598ef492cba66dc6aeabcd4/models/model_normalized.solid.binvox'

with open(vox_file, 'rb') as f:
    exp_vox = binvox_rw.read_as_3d_array(f) # Expected voxels object

# Scale down the voxel model and generate a target voxel tensor
scale_down_factor = 2
max_x, max_y, max_z = exp_vox.data.shape

s = scale_down_factor
x = 0
vx = 0

target_vox_tensor = torch.zeros((int(max_x/s),int(max_y/s),int(max_z/s)))

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

NUM_X, NUM_Y, NUM_Z = target_vox_tensor.shape

grid_sizes = (NUM_X, NUM_Y, NUM_Z)
grid_tensor = torch.zeros((NUM_X,NUM_Y,NUM_Z), dtype=torch.long) # just tracks which cells are occupied

def reset():
    # Initialize current_design_tensor

    current_design_tensor = torch.ones((BLOCK_INFO,NUM_X,NUM_Y,NUM_Z), dtype=torch.long) * -1
    # current_design_tensor = torch.randint(low=-1, high=40, size=(BLOCK_INFO,NUM_X,NUM_Y,NUM_Z), dtype=torch.long)

    current_design_tensor[0,:,:,:] = ShapeNetID

    return current_design_tensor

def no_block_conflict(actions):
    check_cells = actions['occupied_cells']
    n_cells, _ = check_cells.shape
    for i in range(n_cells):
        x,y,z = list(check_cells[i])
        if grid_tensor[x,y,z] == 1:
            return False
    return True

def step(state):
    agent = Agent.CNNAgent(grid_sizes=grid_sizes, num_orient=NUM_ORIENTATION, block_info_size=BLOCK_INFO, block_types=BLOCK_TYPES)

    conflict_check_passed = False
    while (not conflict_check_passed):
        agent_actions = agent.select_actions(state)

        block_type_i, orientation, grid_x, grid_y, grid_z = agent_actions
        block_type = list(BLOCK_DEFINITIONS.keys())[block_type_i]

        if orientation == 0:
            grid_position = np.array([grid_x,grid_y,grid_z])
            occupied_cells = grid_position + BLOCK_DEFINITIONS[block_type]['o0_cells']
        if orientation == 1:
            grid_position = np.array([grid_x,grid_y,grid_z])
            occupied_cells = grid_position + BLOCK_DEFINITIONS[block_type]['o1_cells']

        actions = {
            "block_type": block_type,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z,
            "orientation": orientation,
            "occupied_cells": occupied_cells
        }
        conflict_check_passed = no_block_conflict(actions)

    print("Passed")

if __name__ == "__main__":
    current_design_tensor = reset()
    step(state=current_design_tensor)