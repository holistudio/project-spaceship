import numpy as np
import torch
import binvox_rw

BLOCK_INFO = 7
NUM_ORIENTATION = 2

BLOCK_TYPES = 6

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

# Initialize current_design_tensor

current_design_tensor = torch.ones((BLOCK_INFO,NUM_X,NUM_Y,NUM_Z), dtype=torch.long) * -1

current_design_tensor[0,:,:,:] = ShapeNetID

print(current_design_tensor[:,0,0,0])