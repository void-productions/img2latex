#!/usr/bin/env python

import os
import sys

from img2latex.constants import CHARS, CLASSIFIER_IMAGE_SIZE
import img2latex.classifier as classifier
import img2latex.extract as extract
import img2latex.misc as misc


if len(sys.argv) != 2:
	print("requires one <dataset> argument")
	sys.exit()

classifierdir = os.path.join(misc.resdir(), "classifier")
dataset = extract.extract_yaml_dataset(os.path.join(classifierdir, sys.argv[1]), (*CLASSIFIER_IMAGE_SIZE, 3))

gen = classifier.predict(dataset[0])

for i, k in enumerate(gen):
	prediction = CHARS[k["classes"]]
	expected = CHARS[dataset[1][i]]
	print("predicted: {}, expected: {}, probabilities: {}".format(prediction, expected, k["probabilities"]))
