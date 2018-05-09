#!/usr/bin/env python

import sys
sys.path.append("src")

from constants import CHARS
import classifier
from constants import CLASSIFIER_INPUT_SHAPE
import extract
import os
import misc

if len(sys.argv) != 2:
	print("requires one <dataset> argument")
	sys.exit()

classifierdir = os.path.join(misc.resdir(), "classifier")
dataset = extract.extract_yaml_dataset(os.path.join(classifierdir, sys.argv[1]), (*CLASSIFIER_INPUT_SHAPE, 3))
gen = classifier.predict(dataset[0])

for i, k in enumerate(gen):
	prediction = CHARS[k["classes"]]
	expected = CHARS[dataset[1][i]]
	print("predicted: {}, expected: {}".format(prediction, expected))
