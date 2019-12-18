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
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers.merge import concatenate, add
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from PIL import Image


augments_dict = {
    "Flip Horizontal": iaa.Fliplr(),
    "Flip Vertical": iaa.Flipud(),
    "Blur": iaa.GaussianBlur(sigma=(0.0, 0.5)),
    "Contrast": iaa.LinearContrast(0.5, 1.25),
    "Noise": iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    "Color": iaa.Multiply((0.8, 1.2), per_channel=0.2),
    "Rotate": iaa.Affine(rotate=(-360, 360)), # rotate the image randomly between -360 and 360 degrees
    "Shear": iaa.Affine(shear=(-45, 45)), # shear the image between -45 and 45 degrees
    "Scale": iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}) # random scaling from 80% to 120% of original size
}  
#this method takes in the path of the directory containing images and their respective masks, and returns a 4d array with their respective values
# resizes the images to 256 x 256px
def get_images(path, im_height=256, im_width=256):
    # get directory path for unaugmented images
    #imgs_path = os.path.join(path, "images_unaugmented")

    imgs = os.listdir(path)

    # remove .DS_Store, if it exists in either directory
    if ".DS_Store" in imgs:
        imgs.remove('.DS_Store')

    # remove directory .ipynb_checkpoints if it exists
    if ".ipynb_checkpoints" in imgs:
        shutil.rmtree(os.path.join(path, '.ipynb_checkpoints'))

    imgs = os.listdir(path)

    # arrays are 4d numpy array of shape (N, height, width, channels)
    im_arr = np.zeros((len(imgs), im_height, im_width, 3), dtype=np.uint8)

    for n, id_ in tqdm_notebook(enumerate(imgs), total=len(imgs)):
        # Load images
        img = load_img(os.path.join(path, id_))
        image = img_to_array(img)
        image = resize(image, (256, 256, 3), mode='constant', preserve_range=True)

        im_arr[n] = image
  
    return im_arr

# does random augmentations to the images at the given path 
def random_augment(paths, withmasks=False):
    #check the length of the paths to see whether is multi or single 
    multiple = len(paths) > 1
   
    if multiple:
        # if processing multiple batches
        # returns 2 arrays of images
        if withmasks:
            imgs, masks = rand_aug_mult_wmasks(*paths)
            return imgs, masks
        else:
            imgs_one, imgs_two = rand_aug_mult(*paths)
            return imgs_one, imgs_two
    else:
        # if processing a single batch, there is only one path, get that path
        # only returns a single array of images
        # use *paths[0] to return the string of the path
        if withmasks:
            return rand_aug_single_mask(paths[0])
        else:
            return rand_aug_single(paths[0])

### MANUAL SELECTED AUGMENTATIONS ###
# takes in the paths and the list of options
def manual_augment(paths, options):
    #obtain an array of the augments
    aug_arr = []
    for op in options:
        if op == "yes" or op == "no":
            continue
        else:
            augment = augments_dict[op]
            aug_arr.append(augment)
    
    # use the array of augments to generate an augment sequence
    seq = iaa.Sequential(aug_arr)

    # check if user wants to augment multiple batches
    multiple = len(paths) > 1
    
    if multiple:
        # make sure that the augmentation sequence is deterministic 
        # so that augmentations are same across both batch
        seq = seq.localize_random_state()
        seq = seq.to_deterministic()
        imgs_one, imgs_two = manual_augment_mult(seq, *paths)
        return imgs_one, imgs_two
    else:
        # use *paths[0] to return the string of the path
        return manual_augment_single(paths[0], seq)
    
def manual_augment_mult(paths, seqs):  
    # get the paths for the images
    imgs_one_path = paths[0]
    imgs_two_path = paths[1]

    # get the images from the path
    imgs_one = get_images(imgs_one_path)
    imgs_two = get_images(imgs_two_path)

    # augment the images using the same sequence
    imgs_one_aug = seq.augment_images(imgs_one)
    imgs_two_aug = seq.augment_images(imgs_two)
    
    return imgs_one_aug, imgs_two_aug

def manual_augment_single(path, seq):
    # get the images from the specified path
    imgs = get_images(path)

    # augment and return the images
    return seq.augment_images(imgs)
    
##########################

### RANDOM AUGMENTATIONS ###
# random augmentations for multiple batches, one original, one mask
def rand_aug_mult_wmasks(paths):
    imgs_path = paths[0]
    masks_path = paths[1]

    imgs = get_images(imgs_path)
    masks = get_images(masks_path)

    # sequence for images
    seq_imgs = iaa.Sequential([
        iaa.Fliplr(0.5, random_state=1), # horizontal flips
        iaa.Flipud(0.5, random_state=2), # vertical flips
        iaa.Crop(percent=(0, 0.1), random_state=3), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))),

        # Strengthen or weaken the contrast in each image.
        iaa.contrast.LinearContrast((0.5, 1.25), random_state=4),
        # Add gaussian noise for 50% of the images.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.Sometimes(0.5, 
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), random_state=5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2, random_state=6),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # random scaling from 80% to 120% of original size
            translate_percent={"x": (0.0, 0.2), "y": (0.0, 0.2)},# translation from 0 translation to 20% of axis size
            rotate=(-360, 360), # rotate the image randomly between -360 and 360 degrees
            shear=(-45, 45),
            cval=255,#set cval to 255 to prevent any black areas occuring 
            mode='constant', random_state=7
        )
    ], random_state=8) # apply augmenters in random order

    # Mask-specific sequence
    seq_masks = iaa.Sequential([
        iaa.Fliplr(0.5, random_state=1), # horizontal flips
        iaa.Flipud(0.5, random_state=2), # vertical flips
        iaa.Crop(percent=(0, 0.1), random_state=3), # random crops
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # random scaling from 80% to 120% of original size
            translate_percent={"x": (0.0, 0.2), "y": (0.0, 0.2)},# translation from 0 translation to 20% of axis size
            rotate=(-360, 360), # rotate the image randomly between -360 and 360 degrees
            shear=(-45, 45),
            cval=255,#set cval to 255 to prevent any black areas occuring 
            mode='constant', random_state=7
        )
    ], random_state=8) # apply augmenters in random order

    imgs_aug = seq_imgs.augment_images(imgs)
    masks_aug = seq_masks.augment_images(masks)

    return imgs_aug, masks_aug

# random augmentation for multiple batches of normal images
def rand_aug_mult(paths):
    imgs_one_path = paths[0]
    imgs_two_path = paths[1]

    imgs_one = get_images(imgs_one_path)
    imgs_two = get_images(imgs_two_path)
    #generate a random seed 
    random_seed = np.random.randint(low=1, high=10)
    ia.seed(random_seed)

    # sequence for first batch
    seq_imgs_one = iaa.Sequential([
        iaa.Fliplr(0.5, random_state=1), # horizontal flips
        iaa.Flipud(0.5, random_state=2), # vertical flips
        iaa.Crop(percent=(0, 0.1), random_state=3), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))),

        # Strengthen or weaken the contrast in each image.
        iaa.contrast.LinearContrast((0.5, 1.25), random_state=4),
        # Add gaussian noise for 50% of the images.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.Sometimes(0.5, 
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), random_state=5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2, random_state=6),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # random scaling from 80% to 120% of original size
            translate_percent={"x": (0.0, 0.2), "y": (0.0, 0.2)},# translation from 0 translation to 20% of axis size
            rotate=(-360, 360), # rotate the image randomly between -360 and 360 degrees
            shear=(-45, 45),
            cval=255,#set cval to 255 to prevent any black areas occuring 
            mode='constant', random_state=7
        )
    ], random_state=8) # apply augmenters in random order

    # augmentation sequence for second batch
    seq_imgs_two = iaa.Sequential([
        iaa.Fliplr(0.5, random_state=1), # horizontal flips
        iaa.Flipud(0.5, random_state=2), # vertical flips
        iaa.Crop(percent=(0, 0.1), random_state=3), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))),

        # Strengthen or weaken the contrast in each image.
        iaa.contrast.LinearContrast((0.5, 1.25), random_state=4),
        # Add gaussian noise for 50% of the images.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.Sometimes(0.5, 
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), random_state=5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2, random_state=6),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # random scaling from 80% to 120% of original size
            translate_percent={"x": (0.0, 0.2), "y": (0.0, 0.2)},# translation from 0 translation to 20% of axis size
            rotate=(-360, 360), # rotate the image randomly between -360 and 360 degrees
            shear=(-45, 45),
            cval=255,#set cval to 255 to prevent any black areas occuring 
            mode='constant', random_state=7
        )
    ], random_state=8) # apply augmenters in random order

    #apply the augmentations
    imgs_one_aug = seq_imgs_one.augment_images(imgs_one)
    imgs_two_aug = seq_imgs_one.augment_images(imgs_two)

    return imgs_one_aug, imgs_two_aug

# random augmentation for a single batch of masks
def rand_aug_single_mask(mask_path):
    # get masks from mask path
    masks = get_images(mask_path)
    # generate a random seed 
    random_seed = np.random.randint(low=1, high=10)
    ia.seed(random_seed)

    # Mask-specific sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.5), # vertical flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # random scaling from 80% to 120% of original size
            translate_percent={"x": (0.0, 0.2), "y": (0.0, 0.2)},# translation from 0 translation to 20% of axis size
            rotate=(-360, 360), # rotate the image randomly between -360 and 360 degrees
            shear=(-45, 45),
            cval=255,#set cval to 255 to prevent any black areas occuring 
            mode='constant'
        )
    ], random_order=True)

    augmented_masks = seq.augment_images(masks)

    return augmented_masks

# random augmentation of single batch of normal images
def rand_aug_single(img_path):
    # get the images from the specified path
    imgs = get_images(img_path)
    # generate a random seed 
    random_seed = np.random.randint(low=1, high=10)
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
            iaa.contrast.LinearContrast((0.5, 1.25)),
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

    # apply the augments to the images
    augmented_imgs = seq.augment_images(imgs)

    return augmented_imgs
