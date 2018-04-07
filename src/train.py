#!/usr/bin/env python3

import splitter, classifier, assembler

import tensorflow as tf
import sys
import os

def usage():
	print("train.py <net>")
	sys.exit()

def main():
	if len(sys.argv) != 1:
		usage()

	net = sys.argv[1]
	if net == "classifier":
		classifier.train()
	else:
		print("unknown net")
	

if __name__ == "__main__":
	main()
