import cv2
import numpy as np
import os

from ruamel.yaml import YAML
from ruamel.yaml.scanner import ScannerError

#from data import 


IMAGE_DIRECTORY = "images"
LABEL_FILE = "labels.yaml"
IMAGE_ENDING = ".png"


def image_files_to_array(image_files, number_of_images):
	"""
	Reads the given image files and returns one numpy array filled with all images.
	"""

	width = -1
	height = -1
	depth = 1
	initialized = False

	for index, path in enumerate(image_files):
		if not os.path.isfile(path):
			raise IOError("could not open '{}'".format(path))
		img = cv2.imread(path)
		if not initialized:
			height = img.shape[0]
			width = img.shape[1]
			if len(img.shape) == 3:
				depth = img.shape[2]
			initialized = True
			images = np.empty((number_of_images, height * width * depth))
		images[index] = np.reshape(img, height * width * depth)

	return images

def extract_image_dataset(path):
	"""
	Reads a image-dataset directory and returns a tuple (image_data, label_data)
	A image-dataset directory should contain the directories images/ and labels/
	"""

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
	"""
	pass

def extract_yaml_dataset(path):
	"""
	Reads the yaml-dataset with the specified path and returns a tuple(x_data, y_data)
	A yaml-dataset directory should contain the directories images/ and labels/

	:param path: Path to the yaml-dataset directory
	:type path: str
	"""

	image_directory = os.path.join(path, IMAGE_DIRECTORY)
	yaml_path = os.path.join(path, LABEL_FILE)

	if not os.path.isdir(image_directory):
		raise IOError("could not found image_directory: '{}'".format(image_directory))

	# read labels
	yaml = YAML()
	try:
		with open(yaml_path, "r") as f:
			description = yaml.load(f)
	except IOError:
		raise IOError("Could not open '{}'".format(yaml_path))
	except ScannerError:
		raise IOError("Could not parse '{}'".format(yaml_path))

	number_of_images = len(description)

	labels = list(description.values())

	image_files = map(lambda x: os.path.join(image_directory, x) + IMAGE_ENDING,
			description.keys())

	x_data = image_files_to_array(image_files, number_of_images)

	return (x_data, labels)

