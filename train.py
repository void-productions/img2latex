#!/usr/bin/env python3

import sys
sys.path.append("src")

import classifier
from constants import CLASSIFIER_IMAGE_SIZE
import extract
import os
import misc

if len(sys.argv) != 2:
	print("requires one <dataset> argument")
	sys.exit()

classifierdir = os.path.join(misc.resdir(), "classifier")
dataset = extract.extract_yaml_dataset(os.path.join(classifierdir, sys.argv[1]), (*CLASSIFIER_IMAGE_SIZE, 3))
classifier.train(*dataset)
