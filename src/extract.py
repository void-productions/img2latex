import cv2
import numpy as np
import os

import yaml

def image_files_to_array(file_pathes):
    """
    Reads the given image files and returns one numpy array filled with all images.
    """

    width = -1
    height = -1
    depth = 1
    initialized = False

    index = 0

    for path in file_pathes:
        img = cv2.imread(path)
        if not initialized:
            height = img.shape[0]
            width = img.shape[1]
            if len(img.shape) == 3:
                depth = img.shape[2]
            initialized = True
            images = np.zeros((len(file_pathes), height * width * depth))
        images[index] = np.reshape(img, height * width * depth)
        index += 1

    return images

def extract_data(path):
    """
    Returns the training data in the specified path.
    """

    # get directories in <path>
    l = os.listdir(path)
    l = map(lambda x: os.path.join(path, x), l)
    l = list(filter(lambda x: os.path.isdir(x), l))

    inputs = map(lambda x: os.path.join(x, "in"), l)
    inputs = image_files_to_array(list(inputs))

    if os.path.exists(os.path.join(l[0], "out.yaml")):
        outputs = list(map(lambda x: yaml.load(open(os.path.join(x, "out.yaml"))), l))
        print("outputs=" + str(outputs))
    else:
        outputs = list(map(lambda x: os.path.join(x, "out.png"), l))
        outputs = image_files_to_array(outputs)

    return inputs, outputs
