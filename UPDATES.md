# UPDATES

A log of project progress and specific lessons learned.

## Milestones

- [ ] **Mercury**: A 3D block environment within Unity's GUI that can read a JSON file to display an arrangement of blocks of predefined sizes.
- [ ] **Venus**: A labeled dataset of 3D models of object that are built with blocks of predefined sizes
  - [X] Look into ShapeNet (got access on HuggingFace!)
- [ ] **Earth**: A trained Block Laying Agent
- [ ] **Moon**: Revisit overall approach. Is RL a necessary part of this approach?
- [ ] **Mars**: Build a standalone 3D block environment app
- [ ] **Jupiter**: Use trained Block Laying Agent to pre-train Object Recognition RL Agent
- [ ] **Saturn**: Connect AI Agents to 3D block app
- [ ] **Uranus**: Local user testing
- [ ] **Neptune**: Deploy the 3D block app to the web
- [ ] **Pluto**: Build a spaceship and explore the universe

## Log

### 2024-03-09

Had to carefully think about how blocks rotate and how that affects the cells they occupy in a tensor and the position in Unity.

Rewards should account that there's a lot of blank space 22161 blank vs 39983

Also need to penalize agent when it places a block that overlaps with other existing blocks.

### 2024-02-25

Somehow I need to go from an input tensor of shape (7,100,100,100) (i.e., `block_info, num_x, num_y, num_z`) and generate a tensor of shape (6,2,100,100,100) (i.e., `block_types, num_orientations, num_x, num_y, num_z`) to interpret as the Block Laying Agent's action space.

The simplest architecture seems to be a single Conv3D layer with `in_channels=7` and `out_channels=12` (`block_types*num_orientations`) and `kernel_size=1` and then reshaping the resulting (12,100,100,100) tensor to (6,2,100,100,100). The nice thing is that this generates the action space "all in one shot" that hopefull the neural net "considers everything all at once" when deciding which block type to use, what orientation it has to be in and where to place it in XYZ coordinates.

But I suspect there may be better ways to do it. With respect to the architecture, adding more layers may reflect the hierarchical nature of placing blocks (i.e., "first you find the best block type then figure out where to put it..."). With respect to creating higher dimensional spaces (4D->5D tensor), there might be other ways besides reshaping an oversized 4D tensor.


### 2024-02-25

Played around with the Conv3D function. Assuming an input tensor of (7, 500, 500, 500), it takes awhile for a 3D convolution to finish given the large size and the Conv3D's stride=1. Increasing the kernel size obviously doesn't help because that just increases the amount of computations per "kernel sliding window." Increasing the stride helps reduce the number of computations but at the trade off of reducing the dimensions of the output "too much" - you may lose the complexity in the data after the convolution encodes it to lower dimensional space.

Given that these 3D convolutions can take a long time it might be worth looking at ways to reduce the necessary computations by considering which parts of the input tensor has not changed. Can PyTorch make this comparison between one tensor and a previously processed tensor and just look at the kernels that need re-calculation? The thing is, parts of the 3D grid will remain blank for some time in the initial steps when the Block Laying Agent is just starting to place blocks in a small area of the block


### 2024-02-24

Finished playing around with Conv2D function parameters. Started diagramming how Block Laying Agent could work with dataset. 

A burning question that come up is how neural nets can process multi-modal data, since the Block Laying Agent has to consider both the name of the object and the 3D data. It would be even nicer if both the existing sequence of blocks already laid out and the full 3D grid of filled and unfilled cells.

The name of the object might have to be processed as the ShapeNet ID for now, which is based on WordNet's synset offset (???). This can be easily converted to a string later once I have a dictionary mapping ShapeNet ID to the label.

For now a simple strategy regarding the input data is to "fuse" everything into a single 4D tensor, where the ShapeNet ID is copied throughout the tensor. But I'm wondering if there's a way to handle different modes of input data in their unique data formats at the beginning and then somehow "fuse" them later in the network's latent space or something. I'm betting somehow separate networks start with their own input data but then encode things into a shared latent space.

So the next step is to play around with Conv3D and see how it can work with a 4D tensor which will replicate the 3D grid of the Block Environment, but each cell in that 3D grid contains a 1D vector representation of the block that occupies that grid cell.


### 2024-01-05

Block Environment can now display voxels in Unity, but the binvox file has to be converted to a JSON file first in `voxel-tools/write_json.ipynb`. Working with a birdhouse model for now...

The large scale of the voxel models can be an issue. Right now `scale_down_factor` and `if(count_true >= 3):`  handle scaling down the voxel model. `scale_down_factor` checks if nearby cells are also filled with blocks and `count_true` decides how many total blocks are worth converting into a single blocked in the scaled down version of the voxel model. Will need to check if the same variable values work for other objec models and classes.


### 2024-12-30

Got ShapeNet files! Now I just need to figure out how to read the voxel data and visualize them with Cubes in Unity...