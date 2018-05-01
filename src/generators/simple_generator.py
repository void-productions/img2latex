#!/usr/bin/env python3

from ruamel.yaml import YAML
from ruamel.yaml.scanner import ScannerError

RES_FILE = "./res/test.yaml"

def create_data():
    images = []

    bounding_boxes1 = [{"left": 10, "right":20, "top":30, "bottom":40}, {"left": 50, "right":60, "top":70, "bottom":80}]
    image1 = {"name": "first_image", "bounding_boxes": bounding_boxes1}

    bounding_boxes2 = [{"left": 11, "right":21, "top":31, "bottom":41}, {"left": 51, "right":61, "top":71, "bottom":81}]
    image2 = {"name": "second_image", "bounding_boxes": bounding_boxes2}

    images = [image1, image2]

    return images


def main():
    data = create_data()

    yaml = YAML()
    with open(RES_FILE, "w") as f:
        data = yaml.dump(data, f)

if __name__ == "__main__":
    main()
