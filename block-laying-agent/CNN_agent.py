import torch
import torch.nn as nn

n_h = 64

class CNNQ(nn.Module):
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
        self.DQN = CNNQ(grid_sizes, num_orient, block_info_size, block_types, n_hidden=n_h)

    def select_actions(self, state):
        batch_state = state.unsqueeze(0).float()

        out = self.DQN(batch_state)
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
        return indices

if __name__ == "__main__":
    x = torch.rand((7,100,100,100))
    x = x.unsqueeze(0)

    model = CNNQ(n_hidden=n_h)

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