import torch
import torch.nn as nn

import random
import math

n_h = 64

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
class CNN_DQN(nn.Module):
    def __init__(self, grid_sizes, num_orient, block_info_size, block_types, n_hidden):
        super().__init__()
        self.num_x, self.num_y, self.num_z = grid_sizes
        self.num_orient = num_orient
        self.block_types = block_types
        self.conv1 = nn.Conv3d(in_channels=block_info_size, out_channels=n_hidden, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=n_hidden, out_channels=block_types*num_orient, kernel_size=1)

    def forward(self, x):
        (N, C, D, H, W) = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.reshape(x, (N, self.block_types, self.num_orient, self.num_x, self.num_y, self.num_z))
        return x

class CNNAgent(object):
    def __init__(self, grid_sizes, num_orient, block_info_size, block_types):
        self.action_space = [block_types, num_orient, grid_sizes[0],grid_sizes[1],grid_sizes[2]]

        self.policy_net = CNN_DQN(grid_sizes, num_orient, block_info_size, block_types, n_hidden=n_h)
        self.target_net = CNN_DQN(grid_sizes, num_orient, block_info_size, block_types, n_hidden=n_h)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.steps_done = 0

    def select_actions(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)

        self.steps_done += 1
        print(f'Epsilon={eps_threshold}')
        if sample > eps_threshold:
            print('Agent EXPLOITS!')
            batch_state = state.unsqueeze(0).float()
            with torch.no_grad():
                out = self.policy_net(batch_state)
                out = out.squeeze()

                # Find the index of the maximum value
                max_index = torch.argmax(out)

            # Convert the flat index to multidimensional indices
            indices = []
            for dim_size in reversed(out.shape):
                indices.append((max_index % dim_size).item())
                max_index //= dim_size

            # Reverse the list of indices to match the tensor's shape
            indices.reverse()
        else:
            print('Agent EXPLORES!')
            indices = [random.randint(0,a-1) for a in self.action_space]
        
        agent_actions = torch.tensor(indices, device=device, dtype=torch.long)

        return agent_actions

if __name__ == "__main__":
    x = torch.rand((7,100,100,100))
    x = x.unsqueeze(0)

    model = CNN_DQN(n_hidden=n_h)

    out = model(x)

    out = out.squeeze()

    # Find the index of the maximum value
    max_index = torch.argmax(out)

    # Convert the flat index to multidimensional indices
    indices = []
    for dim_size in reversed(out.shape):
        indices.append((max_index % dim_size).item())
        max_index //= dim_size

    # Reverse the list of indices to match the tensor's shape
    indices.reverse()

    print("Indices of the maximum Q-value:", indices)