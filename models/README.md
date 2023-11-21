# Definition of augemntation methods

- **ZNormalization()**: Subtract mean and divide by standard deviation.
- **PadOrCrop()**: Modify the field of view by cropping or padding to match a target shape. This transform modifies the affine matrix associated to the volume so that physical positions of the voxels are maintained.
- **RandomAffine()**: Apply a random affine transformation and resample the image.

# Methods from the article:
- 9th augmentation method: translation and shearing. The translation step is applied on the x and y-axis with diferent values sampled randomly within the range [−15, 15]. In the shearing step, an angle is selected randomly within the range [−15°, 15°], and shearing is applied within the x and y-axis. The subsequent operations are applied 10 times and 10 images are generated from an input image by this method
  - Translation: 