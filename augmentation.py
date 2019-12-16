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

#this method takes in the path of the directory containing images and their respective masks, and returns a 4d array with their respective values
def get_images(path, im_height=256, im_width=256):
    # get directory path for unaugmented images
    imgs_path = os.path.join(path, "images_unaugmented")

    imgs = os.listdir(imgs_path)

    # remove .DS_Store, if it exists in either directory
    if ".DS_Store" in imgs:
        imgs.remove('.DS_Store')

    # remove directory .ipynb_checkpoints if it exists
    if ".ipynb_checkpoints" in imgs:
        shutil.rmtree(os.path.join(imgs_path, '.ipynb_checkpoints'))

    imgs = os.listdir(imgs_path)

    # arrays are 4d numpy array of shape (N, height, width, channels)
    im_arr = np.zeros((len(imgs), im_height, im_width, 3), dtype=np.uint8)

    for n, id_ in tqdm_notebook(enumerate(imgs), total=len(imgs)):
        # Load images
        img = load_img(os.path.join(imgs_path, id_))
        image = img_to_array(img)
        image = resize(image, (256, 256, 3), mode='constant', preserve_range=True)

        im_arr[n] = image
  
    return im_arr

def random_augment_single(img_arr):
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
    augmented_imgs = seq.augment_images(img_arr)

    # return the augmented img_arr, as a 4D numpy array
    return augmented_imgs

def manual_augment(img_arr, options):
    #obtain an array of the augments
    aug_arr = []
    for op in options:
        augment = augments_dict[op]
        aug_arr.append(augment)

    # use the array of augments to generate an augment sequence
    seq = iaa.Sequential(aug_arr)
    augmented_imgs = seq.augment_images(img_arr)
    return augmented_imgs
    

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

