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