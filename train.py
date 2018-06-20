#!/usr/bin/env python3

import sys
import os

import img2latex.classifier as classifier
from img2latex.constants import CLASSIFIER_INPUT_SHAPE
import img2latex.extract as extract
import img2latex.misc as misc


if len(sys.argv) != 2:
	print("requires one <dataset> argument")
	sys.exit()

classifierdir = os.path.join(misc.resdir(), "classifier")
dataset = extract.extract_yaml_dataset(os.path.join(classifierdir, sys.argv[1]), (*CLASSIFIER_INPUT_SHAPE, 3))
classifier.train(*dataset)
