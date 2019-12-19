
#import augmenter.augmentation as aug
#import augmenter.augment_v2 as aug2
import os
import sys
sys.path.append(os.path.basename(sys.argv[0]))
import augmentation as aug
import json
from PIL import Image

import shutil
import glob
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import random

#import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#from PIL import Image
# Start initializes the process. Checks if there are multiple folders
def start(dict):
    """dict should contain only the fields in the exact format:
        {"folder":[...],
         "mode":[<Either "Random" or "Manual"],
         "options":[<Any one of: 
            "Flip Horizontal"
            "Flip Vertical"
            "Blur"
            "Contrast"
            "Noise"
            "Color"
            "Rotate"
            "Shear"
            "Scale"
            >] }
    """
    # takes in options (a dictionary) from the user, and a 4d numpy arr of images, imgs, to be augmented
    # mode is the mode that user selects, either random or manual
    folder_list = dict["folder"]

    mode_list = dict["mode"]

    options_list = dict["options"]

    augmented_images = []
    if len(folder_list) == 1:
        # Go to the next stage
        # If the length of the folder list is 1, means that there is only a single folder to process
        print("Start processing for single batch")
        augmented_images = process_single(folder_list[0], mode_list, options_list)
        save_path = "./save_test"
        save_single(augmented_images, save_path, folder_list)
    else: 
        print("Start processing for multiple batch")
        augmented_images_1, augmented_images_2 = process_multiple(folder_list, mode_list, options_list)
        augmented_images = [augmented_images_1, augmented_images_2]
        save_path1 = "./save_mult_test1"
        save_path2 = "./save_mult_test2"
        save_paths = [save_path1,save_path2]
        save_multiple(augmented_images, save_paths, folder_list)

# Processes a single batch of images
def process_single(folder_path, mode_list, options_list):
    # Next stage, check if random or manual
    mode = mode_list[0]
    # check if random or manual
    if mode == "Random":
        print("processing single batch (RANDOM)")
        return aug.single_random(folder_path)
    else:
        # Mode is manual
        print("processing single batch (MANUAL)")
        return aug.single_manual(folder_path, options_list)

# Processes a multiple (2) batches of images
def process_multiple(folder_list, mode_list, options_list):
    # check if its random or manual
    mode = mode_list[0]
    if mode == "Random":
        print("processing multiple batch (RANDOM)")
        #return aug.multiple_random(folder_list)
        return aug.multiple_random_mask(folder_list)
    else: 
        print("processing multiple batch (MANUAL)")
        #return aug.multiple_manual(folder_list, options_list)
        return aug.multiple_manual_mask(folder_list, options_list)

def save_single(imgs_arr, save_path, original_path):
    print("Saving images...")
    save_dir = sorted(os.listdir(save_path))
    if ".DS_Store" in save_dir:
        save_dir.remove(".DS_Store")

    orig_dir = sorted(os.listdir(original_path[0]))
    if ".DS_Store" in orig_dir:
        orig_dir.remove('.DS_Store')

    for i in range(len(imgs_arr)):
        augmented_img = Image.fromarray(imgs_arr[i], "RGB")
        # obtain original name from the original directory
        original_name = os.path.splitext(orig_dir[i])
        # concat and generate new name
        new_name = original_name[0] + "_augmented" + original_name[1]
        # save the image w the new name to the save path
        augmented_img.save(os.path.join(save_path, new_name))
    print("Save successful")

# Params: 
# imgs: list of size 2, each elem (imgs[i]) is a 4d numpy arr containing the images for each dir
# save_paths: list of size 2, each elem is the path to save the corresponding imgs[i] to
# original_paths: list of size 2, each elem is the original path of the imgs[i]
def save_multiple(imgs, save_paths, original_paths):
    #get the corresponding variables
    imgs_first = imgs[0] #4d array representing the images of the first folder
    imgs_second = imgs[1] #4d array representing the images of the second folder

    save_path_first = save_paths[0] # string representing path to the first save destination folder
    save_path_second = save_paths[1] # string representing path to the second save destination folder

    save_dir_1 = sorted(os.listdir(save_path_first))
    if ".DS_Store" in save_dir_1:
        save_dir_1.remove(".DS_Store")

    save_dir_2 = sorted(os.listdir(save_path_second))
    if ".DS_Store" in save_dir_2:
        save_dir_2.remove(".DS_Store")

    orig_path_first = original_paths[0] #string representing the path to the first original folder
    orig_path_second = original_paths[1] #string representing the path to the second original folder
    # save the first directory 
    print("Saving images for first directory...")
    orig_dir_first = sorted(os.listdir(orig_path_first)) # list the items in the directory to
    if ".DS_Store" in orig_dir_first:
        orig_dir_first.remove('.DS_Store')


    for i in range(len(imgs_first)):
        augmented_img = Image.fromarray(imgs_first[i], "RGB")
        # get the original name from the directory
        original_name = os.path.splitext(orig_dir_first[i])
        # concat and generate new name
        new_name = original_name[0] + "_augmented" + original_name[1]
        # save to the save path
        augmented_img.save(os.path.join(save_path_first, new_name))
    print("Save successful for first")

    # save for second directory
    print("Saving images for second directory...")
    orig_dir_second = sorted(os.listdir(orig_path_second))
    if ".DS_Store" in orig_dir_second:
        orig_dir_second.remove('.DS_Store')

    for i in range(len(imgs_second)):
        augmented_img = Image.fromarray(imgs_second[i], "RGB")
        # get the original name from the directory list
        original_name = os.path.splitext(orig_dir_second[i])
        new_name = original_name[0] + "_augmented" + original_name[1]
        augmented_img.save(os.path.join(save_path_second, new_name))
    print("Save successful for second")


# lines below were used for testing
given_dict = json.loads(sys.argv[1])
start(given_dict)