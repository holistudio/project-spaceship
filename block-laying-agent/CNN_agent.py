import torch
import torch.nn as nn

NUM_X = 100
NUM_Y = 100
NUM_Z = 100
BLOCK_INFO = 7
NUM_ORIENTATION = 2

BLOCK_TYPES = 6

n_h = 64

class CNNAgent(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=BLOCK_INFO, out_channels=n_hidden, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=n_hidden, out_channels=BLOCK_TYPES*NUM_ORIENTATION, kernel_size=1)
    def forward(self, x):
        (N, C, D, H, W) = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.reshape(x, (N, BLOCK_TYPES, NUM_ORIENTATION, NUM_Z, NUM_Y, NUM_X))
        return x

x = torch.rand((BLOCK_INFO,NUM_X,NUM_Y,NUM_Z))
x = x.unsqueeze(0)

model = CNNAgent(n_hidden=n_h)

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