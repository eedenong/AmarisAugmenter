Amaris Image Augmentation Script V1.0
================================
#### Introduction ####
Simple script for augmenting images using methods from __imgaug__ library
Supports the following augmentations:
* Horizontal and vertical flips
* Crops
* Linear Contrast
* Gaussian Noise addition
* Gaussian Blur
* Pixel Multiplication (colour change)
* Affine transformations: scale, translate, rotate, shear

#### Use ####
This script is meant to be used with a web interface _(to be added)_


#### Limitations ####
The given path should contain __only__ images in the _.png_ or _.jpeg_ format.

Only two batches of images can be augmented at the same time

If augmenting masks, must make sure that the path to the masks are the second path that is uploaded

Manual augmentation on multiple batches will result in the chosen augmentations to be applied on __both__ the batches

If you want to manually augment multiple batches differently, augment the batches one by one
