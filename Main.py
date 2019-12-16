import augmentation as aug
import sys

def start(selected_options, imgs):
    # takes in options (a dictionary) from the user, and a 4d numpy arr of images, imgs, to be augmented
    # mode is the mode that user selects, either random or manual
    mode = selected_options["mode"]
    batch = selected_options["batch"]
    if mode[0] == "manual":
        # if user chose mode to be manual
        # then options is a list of the chosen augmentation options the user wants
        options = selected_options["options"]
        try: 
            if len(options) == 0:
                #if the user somehow doesnt pass in 
                raise Exception('Please select augmentation options')
            else:
                return process_images(mode, imgs, options)
        except Exception as e:
            print(e)
    else:
        return process_images(mode, imgs)
 

def process_images(mode, imgs, *options):
    random = mode == "random"

    if random and len(options) > 0:
        #get options, which is a list of options
        return aug.manual_augment(imgs, options)
    else:
        # not random, user has to select options
        return aug.random_augment(imgs)


