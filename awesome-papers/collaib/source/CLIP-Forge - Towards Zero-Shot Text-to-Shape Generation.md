
**Link:** https://arxiv.org/pdf/2110.02624.pdf

**Authors**: ...

## Look Into
 - normalizing flow network
 - image-text joint embedding models
 - avoiding the expensive inference time optimizations as employed in existing 2D approaches (citations 18, 57)

## Problem

Generating 3D shape from text input is needed in design, animation, and manufacturing domains.

### Specific Challenges

 - scarcity of text-3D shape dataset
 - expensive inference time optimization

## Related Works

 - ...
 - ...
## Approach

**Dataset:** ShapeNet v2
- 13 rigid object classes
- Rendered images
- Voxel grids
- Query points
- Occupancy

**Training**
1. Render 3D shapes into images
2. Use pre-trained image-text joint embedding models to estimate text associated with those images of 3D shapes. Joint embedding models consist of image and text encoders.
3. Obtain latent space of 3D shapes (i.e., 3D embeddings) by training an autoencoder. Each 3D shape now has a corresponding 1) 3D embedding, 2) text embedding, and 3) image embedding
4. Train a normalizing flow network to relate 3D embeddings to image-text embeddings. Specifically, "model distribution of shape embeddings conditioned on the image features" from the image-text joint embedding models.

**Inference**
1. Given text input, obtain text features using text encoder from image-text joint embedding
2. Generate shape embedding using normalizing flow network.
3. Generate 3D shape by inputing shape embedding into trained autoencoder (i.e. decode shape embedding)

## Key Findings

 - ...
 - ...
 - ...

## Takeaways

 - ...
 - ...
 - ..





