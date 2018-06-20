#!/usr/bin/env python3

import matplotlib.pyplot as plt
from io import BytesIO
import os
from ruamel.yaml import YAML

array = ["a", "b"]
fonts = ["DejaVu Sans"]

def generate_latex_file(x, filename, fontsize=12, dpi=300, font="DejaVu Sans"):
	format_=  "png"

	fig = plt.figure(figsize=(0.01, 0.01))
	fig.text(0, 0, u'{}'.format(x), fontsize=fontsize, fontdict={'family': font})
	buffer_ = BytesIO()
	fig.savefig(buffer_, dpi=dpi, transparent=False, format=format_, bbox_inches='tight', pad_inches=0.0)
	plt.close(fig)
	with open(filename + ".png", "bw") as f:
		f.write(buffer_.getvalue())

dirpath = os.path.dirname(os.path.realpath(__file__)) + "/../../res/classifier/latex/"
images_dirpath = os.path.join(dirpath, "images")

d = dict()

for x in array:
	for font in fonts:
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		if not os.path.exists(images_dirpath):
			os.mkdir(images_dirpath)
		f = "-".join(["example", x, font.replace(" ", "-")])
		filename = os.path.join(images_dirpath, f)
		d[f] = x
		if not os.path.exists(filename):
			generate_latex_file(x, filename, font=font)

yaml = YAML()
with open(dirpath + "labels.yaml", "w") as f:
    yaml.dump(d, f)
