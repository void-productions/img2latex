#!/usr/bin/env python3

import classifier
import extract
import sys
import os
import misc

if len(sys.argv) != 2:
	print("requires one <dataset> argument")
	sys.exit()

classifierdir = os.path.join(misc.resdir(), "classifier")
dataset = extract.extract_yaml_dataset(os.path.join(classifierdir, sys.argv[1]), (50, 50, 3))
classifier.train(*dataset)
