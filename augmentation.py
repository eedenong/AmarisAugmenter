# Script for augmenting images
# - Horizontal and vertical flips
# - Crops
# - Linear Contrast changing
# - Gaussian Noise addition
# - Gaussian Blur
# - Pixel multiplication (colour change)
# - Affine transformations: scale, translate, rotate, shear)
import os
import shutil
import glob
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import random

from skimage.transform import resize

#import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from PIL import Image

SEED_MAX = 1000
augments_dict = {
    "Flip Horizontal": iaa.Fliplr(0.5),
    "Flip Vertical": iaa.Flipud(0.5),
    "Blur": iaa.GaussianBlur(sigma=(0.0, 0.5)),
    "Contrast": iaa.LinearContrast(0.5, 1.0),
    "Noise": iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    "Color": iaa.Multiply((0.8, 1.2), per_channel=0.2),
    "Rotate": iaa.Affine(rotate=(-360, 360), mode="constant"), # rotate the image randomly between -360 and 360 degrees
    "Shear": iaa.Affine(shear=(-45, 45), mode="constant"), # shear the image between -45 and 45 degrees
    "Scale": iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, mode="constant") # random scaling from 80% to 120% of original size
}  
# this method takes in the path of the directory containing images and their respective masks, and returns a 4d array with their respective values
# resizes the images to 256 x 256px
def get_images(path, im_height=256, im_width=256):
    # get directory path for unaugmented images
    #imgs_path = os.path.join(path, "images_unaugmented")
    print("Getting images from " + path)
    imgs = os.listdir(path)

    # remove .DS_Store, if it exists in either directory
    if ".DS_Store" in imgs:
        imgs.remove('.DS_Store')

    # remove directory .ipynb_checkpoints if it exists
    if ".ipynb_checkpoints" in imgs:
        shutil.rmtree(os.path.join(path, '.ipynb_checkpoints'))

    imgs = os.listdir(path)
    imgs.sort()
    # arrays are 4d numpy array of shape (N, height, width, channels)
    im_arr = np.zeros((len(imgs), im_height, im_width, 3), dtype=np.uint8)

    i = 0
    for name in imgs:
        img = load_img(path + name)
        image = img_to_array(img)
        image = resize(image, (256, 256, 3), mode='constant', preserve_range=True)

        im_arr[i] = image
        i += 1
    
    print("Done getting images from " + path)
    return im_arr

#################################
#   SINGLE BATCH AUGMENTATION   #
#################################
# Defines methods to be used by process_single in Main.py

# TESTED OK
def single_random(path):
    # Get images from specified path, as a 4D numpy array
    print("starting SINGLE batch RANDOM augment")
    imgs_arr = get_images(path)
    aug_seq = gen_random_augment()
    imgs_aug = aug_seq.augment_images(imgs_arr)
    str = "successfully augmented img_arr. Final aug shape is {}"
    print(str.format(imgs_aug.shape))
    return imgs_aug

# TESTED OK
def single_manual(path, options):
    print("starting SINGLE batch MANUAL augment")
    imgs_arr = get_images(path)
    aug_seq = gen_manual_augment(options)
    imgs_aug = aug_seq.augment_images(imgs_arr)
    return imgs_aug
#################################
#   MULTI BATCH AUGMENTATION    #
#################################
def multiple_random(folder_list):
    print("starting MULTIPLE batch RANDOM augment")
    imgs_arr_1 = get_images(folder_list[0])
    imgs_arr_2 = get_images(folder_list[1])
    aug_seq = gen_random_augment()
    #aug_seq_2 = iaa.meta.copy_random_state(aug_seq_1)
    imgs_aug_1 = aug_seq.augment_images(imgs_arr_1)
    imgs_aug_2 = aug_seq.augment_images(imgs_arr_2)
    return imgs_aug_1, imgs_aug_2

def multiple_manual(folder_list, options_list):
    print("starting MULTIPLE batch MANUAL augment")
    imgs_arr_1 = get_images(folder_list[0])
    imgs_arr_2 = get_images(folder_list[1])
    aug_seq = gen_manual_augment(options_list)
    #aug_seq_2 = iaa.meta.copy_random_state(aug_seq_1)
    imgs_aug_1 = aug_seq.augment_images(imgs_arr_1)
    imgs_aug_2 = aug_seq.augment_images(imgs_arr_2)
    return imgs_aug_1, imgs_aug_2
#####################################
#   AUGMENTATION HELPER FUNCTIONS   #
#####################################
def gen_random_augment():
    print("Generating random augment...")
    # generate a random seed 
    global SEED_MAX
    random_seed = np.random.randint(low=1, high=SEED_MAX)
    ia.seed(random_seed)
    # generate random sequence of random augments
    seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Flipud(0.5), # vertical flips
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))),

            # Strengthen or weaken the contrast in each image.
            iaa.contrast.LinearContrast((0.5, 1.00)),
            # Add gaussian noise for 50% of the images.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.Sometimes(0.5, 
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # random scaling from 80% to 120% of original size
                translate_percent={"x": (0.0, 0.2), "y": (0.0, 0.2)},# translation from 0 translation to 20% of axis size
                rotate=(-360, 360), # rotate the image randomly between -360 and 360 degrees
                shear=(-45, 45),
                cval=255,#set cval to 255 to prevent any black areas occuring 
                mode='constant')
        ], random_order=True) # apply augmenters in random order

    seq_i = seq.localize_random_state()
    seq_img = seq_i.to_deterministic()
    print("Random augment sequence generated")
    print(seq_img)
    return seq_img

def gen_manual_augment(options):
    print("Generating manual augment...")
    ia.seed(1)
    # generate sequence of augmentations based on the chosen options
    chosen_augments = []
    for opt in options:
        augment_option = augments_dict[opt]
        chosen_augments.append(augment_option)
    
    seq = iaa.Sequential(chosen_augments, random_order=True)

    seq_i = seq.localize_random_state()
    seq_img = seq_i.to_deterministic()

    print("Manual augment generated")
    print(seq_img)
    return seq_img

