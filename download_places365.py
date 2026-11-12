"""
One-time helper that downloads the pretrained Places365 (ResNet18) weights
and the 365-category label file used by scene_classifier.py.

Run this once, from the project root, before starting main.py:

    python download_places365.py
"""

import os
import urllib.request

import config

WEIGHTS_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
CATEGORIES_URL = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"Already exists, skipping: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print("Done.")


if __name__ == "__main__":
    download(WEIGHTS_URL, config.SCENE_MODEL_WEIGHTS)
    download(CATEGORIES_URL, config.SCENE_CATEGORIES_FILE)
    print("\nPlaces365 model ready. You can now run: python main.py")
