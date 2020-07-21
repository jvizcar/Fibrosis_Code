"""
Train a new model on a given dataset, created in grid_tiling_datasets.py
"""
from modules.keras2 import vgg19
import configs as cf

from os import getcwd
from os.path import join

# variables for run
class_names = cf.CLASS_NAMES

# train the model
model = vgg19(join(getcwd(), 'Images'), class_names=class_names)

# predict on training and validation datasets to generate figures
model.predict(dataset='Train')
model.predict(dataset='Val')
model.predict(dataset='Test')

