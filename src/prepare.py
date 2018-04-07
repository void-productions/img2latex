import cv2
import numpy as np
import os

def image_files_to_array(files):
	pass

def prepare_data(path):
	l = os.listdir(path)
	l = map(lambda x: os.path.join(path, x), l)
	l = list(filter(lambda x: os.path.isdir(x), l))

	inputs = map(lambda x: os.path.join(x, "in"), l)
	inputs = image_files_to_array(inputs)

	if os.path.exists(os.path.join(l[0], "out.yaml")):
		outputs = map(lambda x: os.path.join(x, "out.yaml"), l)
		outputs = list(outputs)
	else:
		outputs = map(lambda x: os.path.join(x, "out.png"), l)
		outputs = image_files_to_array(outputs)

	return inputs, outputs

print(prepare_data("./res/classifier/train"))
