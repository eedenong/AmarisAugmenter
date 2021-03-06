Amaris Image Augmentation Script V1.0
================================
#### Introduction ####
Simple script for augmenting images using methods from __imgaug__ library

Supports single batch augmentation of images (for original image dataset augmentation), or multiple batch augmentation of images (to augment original images and their masks at the same time)

Supports the following augmentations:
* Horizontal and vertical flips
* Crops
* Linear Contrast
* Gaussian Noise addition
* Gaussian Blur
* Pixel Multiplication (colour change)
* Affine transformations: scale, translate, rotate, shear

Images will automatically be resized to 256 x 256px 

#### Use ####
Make sure that:
1. __Main.py__ and __augmentation.py__ are in the same directory before running
2. The folders containing the images to be augmented __ONLY__ have images (no sub-folders)
* If masks are to be augmented, make sure that the directory to the masks folder is the __second__ folder!

To run this script in the command line, uncomment the lines:
* given_dict = json.loads(sys.argv[1])
* start(given_dict)

Then do _python Main.py (dictionary)_ , where dictionary is to be of the format that can be found by
opening the python REPL and running:
1. import Main
2. help(Main)



#### Limitations ####
The given path should contain __only__ images in the _.png_ or _.jpeg_ format.

Only two batches of images can be augmented at the same time

If augmenting masks, must make sure that the path to the masks are the second path that is uploaded

Manual augmentation on multiple batches will result in the chosen augmentations to be applied on __both__ the batches

If you want to manually augment multiple batches differently, augment the batches one by one

#### Current Bugs to be fixed ####
- [x] Files are not being saved in the correct order
- [x] Augmentation does not work for __manual__ augments with __multiple__ batches