"""
Create the local dataset of images to train a glomeruli detection CNN.
"""
from modules.girder_utils import login
from modules.grid_tiling import grid_tiling

from os import getcwd
from os.path import join

from pandas import read_csv
from tqdm import tqdm
import configs as cf
import numpy as np
np.warnings.filterwarnings('ignore')


def create_datasets(save_magnification, save_dir):
    """internal function for creating the train, val, and testing datasets, with the ability to modify some aspects
    of the dataset creation.
    Parameters
    ----------
    save_magnification : float
        magnification to save images to, note that regardless of this parameter the output images will be of the same
        size (224 by 224 by default).
    save_dir : str
        directory to create Train, Val, and Test directories.
    """
    for idx, row in tqdm(dataset_info.iterrows(), total=dataset_info.shape[0]):
        item_id = row['item_id']
        dataset = row['dataset']

        if dataset == 'Train':
            grid_tiling(gc, item_id, group_names, join(save_dir, dataset), save_mag=save_magnification,
                        tile_size=cf.TILE_SIZE)
        elif dataset == 'Val':
            grid_tiling(gc, item_id, group_names, join(save_dir, dataset), save_mag=save_magnification,
                        tile_size=cf.TILE_SIZE)
        elif dataset == 'Test':
            grid_tiling(gc, item_id, group_names, join(save_dir, dataset), save_mag=save_magnification, is_test=True,
                        tile_size=cf.TILE_SIZE)


# set variables
mag = cf.MAG

group_names = cf.GLOM_GROUPS
gc = login(dsa='Transplant')
dataset_info = read_csv('DataFiles/data.csv')

# Create the first dataset with default parameters and save magnificatin of 10
create_datasets(mag, join(getcwd(), 'ImagesTest'))
