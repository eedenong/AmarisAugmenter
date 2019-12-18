import augmenter.augmentation as aug
import sys

def start(dict):
    # takes in options (a dictionary) from the user, and a 4d numpy arr of images, imgs, to be augmented
    # mode is the mode that user selects, either random or manual
    folder_list = dict["folder"]

    mode_list = dict["mode"]

    options_list = dict["options"]

    if len(folder_list) == 1:
        # single
        folder_path = folder_list[0]
        # check mode random or manual
        if mode_list[0] == "random":
            #do single random
            return aug.rand_aug_single(folder_path)
        else:
            # do single manual
            return aug.manual_augment_single(folder_path, options_list)
    else:
        # multiple
        pass

def process_images(mode, paths, options):
    random = mode == "random"
    augment_masks = False
    if len(options) > 0:
        augment_masks = options[-1] == "yes"
    # list idx subject to change according to the mask flag
    # if random, call function for random augmentation
    if random:
        if augment_masks:
            return aug.random_augment(paths, withmasks=True)
        else:
            return aug.random_augment(paths, withmasks=False)
    else:
        # not random, user has to select options
        # does not matter if user is augmenting masks or not, up to user discretion
        return aug.manual_augment(options, paths)
'''
def start(selected_options, paths):
    # takes in options (a dictionary) from the user, and a 4d numpy arr of images, imgs, to be augmented
    # mode is the mode that user selects, either random or manual
    mode = selected_options["mode"]

    options = selected_options["options"]

    if mode[0] == "manual":
        # if user chose mode to be manual
        # then options is a list of the chosen augmentation options the user wants
        
        try: 
            if len(options) == 0:
                #if the user somehow doesnt pass in 
                raise Exception('Please select augmentation options')
            else:
                return process_images(mode, paths, options)
        except Exception as e:
            print(e)
    else:
        return process_images(mode, paths, options)


def process_images(mode, paths, options):
    random = mode == "random"
    augment_masks = False
    if len(options) > 0:
        augment_masks = options[-1] == "yes"
    # list idx subject to change according to the mask flag
    # if random, call function for random augmentation
    if random:
        if augment_masks:
            return aug.random_augment(paths, withmasks=True)
        else:
            return aug.random_augment(paths, withmasks=False)
    else:
        # not random, user has to select options
        # does not matter if user is augmenting masks or not, up to user discretion
        return aug.manual_augment(options, paths)
'''