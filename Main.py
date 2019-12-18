import augmenter.augmentation as aug
import sys

def start(selected_options, paths):
    # takes in options (a dictionary) from the user, and a 4d numpy arr of images, imgs, to be augmented
    # mode is the mode that user selects, either random or manual
    mode = selected_options["mode"]

    if mode[0] == "manual":
        # if user chose mode to be manual
        # then options is a list of the chosen augmentation options the user wants
        options = selected_options["options"]
        try: 
            if len(options) == 0:
                #if the user somehow doesnt pass in 
                raise Exception('Please select augmentation options')
            else:
                return process_images(mode, paths, options)
        except Exception as e:
            print(e)
    else:
        return process_images(mode, paths)


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
        