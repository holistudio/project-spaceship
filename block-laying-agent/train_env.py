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

NUM_EPISODES = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ShapeNetID as integer
ShapeNetID = int('02843684')

# Load voxel model
vox_file = '../../datasets/shapenet/02843684/1b73f96cf598ef492cba66dc6aeabcd4/models/model_normalized.solid.binvox'

with open(vox_file, 'rb') as f:
    print('=LOADING VOXEL MODEL=')
    exp_vox = binvox_rw.read_as_3d_array(f) # Expected voxels object

# Scale down the voxel model and generate a target voxel tensor
scale_down_factor = 2
max_x, max_y, max_z = exp_vox.data.shape

s = scale_down_factor
x = 0
vx = 0

target_vox_tensor = torch.zeros((int(max_x/s),int(max_y/s),int(max_z/s)), device=device)

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

filled = 0
unfilled = 0
for x in range(NUM_X):
    for y in range(NUM_Y):
        for z in range(NUM_Z):
            if target_vox_tensor[x,y,z] == 1:
                filled += 1
            else:
                unfilled += 1

# print(NUM_X*NUM_Y*NUM_Z)
print(f'Model Filled Percent = {filled*100/(NUM_X*NUM_Y*NUM_Z):.2f}%')
print()
# print(unfilled)

grid_sizes = (NUM_X, NUM_Y, NUM_Z)
grid_tensor = torch.zeros((NUM_X,NUM_Y,NUM_Z), dtype=torch.long, device=device) # just tracks which cells are occupied

block_seq_index = 0

def reset():
    # Initialize design_tensor

    design_tensor = torch.ones((BLOCK_INFO,NUM_X,NUM_Y,NUM_Z), dtype=torch.long, device=device) * -1
    # design_tensor = torch.randint(low=-1, high=40, size=(BLOCK_INFO,NUM_X,NUM_Y,NUM_Z), dtype=torch.long)

    design_tensor[0,:,:,:] = ShapeNetID

    block_seq_index = 0 

    return design_tensor, 0, False, block_seq_index

def no_block_conflict(actions):
    check_cells = actions['occupied_cells']
    n_cells, _ = check_cells.shape
    for i in range(n_cells):
        x,y,z = list(check_cells[i])
        if (x>=grid_sizes[0]) or (y>=grid_sizes[1]) or (z>=grid_sizes[2]):
            # print(f'! Block Out of Bounds at {x,y,z} !')
            return False
        if (x<0) or (y<0) or (z<0):
            # print(f'! Block Out of Bounds at {x,y,z} !')
            return False
        if grid_tensor[x,y,z] == 1:
            # print(f'! Block Conflict at {x,y,z} !')
            return False
    return True

def add_block(actions, design_tensor):
    pos_x, pos_y, pos_z = actions["grid_position"]
    occupied_cells = actions['occupied_cells']
    n_cells, _ = occupied_cells.shape
    for i in range(n_cells):
        x,y,z = list(occupied_cells[i])
        design_tensor[:,x,y,z] = torch.tensor([ShapeNetID, actions["block_type_i"], pos_x, pos_y, pos_z,
                                               actions['orientation'], block_seq_index], dtype=torch.long, device=device)
        grid_tensor[x,y,z] = 1
    return design_tensor

def calc_reward(diff_tensor):
    rew = 0
    
    # value of 0 means either true positive (block where should be a block) or true negative (blank where should be blank)
    zero_indices = torch.nonzero(diff_tensor == 0)
    target_values = target_vox_tensor[zero_indices[:, 0], zero_indices[:, 1], zero_indices[:, 2]]
    fill_mask = (target_values == 1)
    n_fill = torch.sum(fill_mask).item()
    n_empty = len(zero_indices) - n_fill
    rew += n_fill
    rew += n_empty * 0.1

    # value of -1 means false positive (block placed by agent but should be blank)
    neg_1_mask = (diff_tensor == -1)
    n_fp = torch.sum(neg_1_mask).item()
    rew -= n_fp

    # value of +1 means false negative (should block but none placed by agent)
    pos_1_mask = (diff_tensor == 1)
    n_fn = torch.sum(pos_1_mask).item()
    rew -= n_fn

    perc_complete = n_fill/filled
    return rew, perc_complete

def determine_terminal(diff_tensor, block_seq_index):
    if torch.all(diff_tensor == 0):
        return True
    # if block_seq_index > 10:
    if block_seq_index > filled:
        print('! Number of moves exceeded !')
        return True
    return False

def step(state, agent_actions, block_seq_index):
    env_actions = list(agent_actions.cpu().squeeze().numpy())
    block_type_i, orientation, grid_x, grid_y, grid_z = env_actions
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

    if (no_block_conflict(actions)):
        # print(f'Agent places {block_type} block at {grid_position}, orientation={orientation}')
        next_state = add_block(actions, state)
    else:
        next_state = state

    block_seq_index += 1

    diff_tensor = target_vox_tensor - grid_tensor

    reward, perc_complete = calc_reward(diff_tensor)
    
    if (not no_block_conflict(actions)):
        block_conflict_penalty = -100000
        reward = block_conflict_penalty
    
    # print(f'Reward = {reward}')
    if block_seq_index % 100 == 0:
        print(f'Block {block_seq_index}, Percent complete = {perc_complete*100:.2f}%')

    terminal = determine_terminal(diff_tensor, block_seq_index)

    return next_state, reward, terminal, block_seq_index
    
    

if __name__ == "__main__":
    agent = Agent.CNNAgent(grid_sizes=grid_sizes, num_orient=NUM_ORIENTATION, block_info_size=BLOCK_INFO, block_types=BLOCK_TYPES)

    for ep in range(NUM_EPISODES):
        print(f'==EPISODE {ep}==')

        state, reward, terminal, block_seq_index = reset()

        while not terminal:
            agent_actions = agent.select_actions(state)

            next_state, reward, terminal, block_seq_index = step(state, agent_actions, block_seq_index)

            agent.update_experience(state,agent_actions,next_state,reward,terminal)

            state = next_state
        print(f'==END EPISODE {ep}==')

        agent.update_policy(ep)
        