# Definition of augemntation methods

- **ZNormalization()**: Subtract mean and divide by standard deviation.
- **PadOrCrop()**: Modify the field of view by cropping or padding to match a target shape. This transform modifies the affine matrix associated to the volume so that physical positions of the voxels are maintained.
- **RandomAffine()**: Apply a random affine transformation and resample the image.