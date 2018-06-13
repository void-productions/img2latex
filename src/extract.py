import cv2
import numpy as np
import os

from constants import CHARS_REV

from ruamel.yaml import YAML
from ruamel.yaml.scanner import ScannerError

IMAGE_DIRECTORY = "images"
LABEL_FILE = "labels.yaml"
IMAGE_ENDING = ".png"


def rectify_image(image):
	"""
	Adds padding to rectify the image
	"""

	height = image.shape[0]
	width = image.shape[1]
	depth = image.shape[2]

	maxi = max(width, height)

	rectified_image = np.ones((maxi, maxi, depth))

	rectified_image[:height,:width] = image

	return rectified_image



def image_files_to_array(image_files, number_of_images, image_shape):
	"""
	Reads the given image files and returns one numpy array filled with all images.
	"""

	height = image_shape[0]
	width = image_shape[1]
	depth = image_shape[2]

	if depth == 3:
		color_mode = 1 # colored image
	elif depth == 1:
		color_mode = 0 # grayscale image
	else:
		raise ValueError("image_shape[2] (depth) has an invalid value: {}".format(depth))

	images = np.empty((number_of_images, height, width, depth), dtype=np.float32)

	for index, path in enumerate(image_files):
		if not os.path.isfile(path):
			raise IOError("could not open '{}'".format(path))
		img = cv2.imread(path, color_mode)
		img = rectify_image(img)
		img = cv2.resize(img, dsize=image_shape[:2])
		images[index] = np.reshape(img, (height, width, depth))

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

def extract_yaml_dataset(path, image_shape):
	"""
	Reads the yaml-dataset with the specified path and returns a tuple(x_data, y_data)
	A yaml-dataset directory should contain the directories images/ and labels/

	:param path: Path to the yaml-dataset directory
	:type path: str
	:param image_shape: The forced image shape of the x_data
	:type image_shape: tuple(height, width)
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

	labels = np.array(list(map(lambda x: CHARS_REV[x], description.values())))

	image_files = map(lambda x: os.path.join(image_directory, x) + IMAGE_ENDING,
			description.keys())

	x_data = image_files_to_array(image_files, number_of_images, image_shape)
	x_data = ((x_data * 2) / np.max(x_data)) - 1

	return (x_data, labels)
