"""
Create the csv files that are used in the various scripts to know which images to pull from DSA.

PAS images are included.
"""
from modules.girder_utils import login, get_recursive_items
from modules.annotation_utils import annotation_info

from pandas import DataFrame
import configs as cf
from tqdm import tqdm

from os.path import join
from os import getcwd, makedirs

# authenticate client
gc = login(dsa='Transplant')

# create directory for data files
cwd = getcwd()
datafiles_dir = join(cwd, 'DataFiles')
makedirs(datafiles_dir, exist_ok=True)

data = {'name': [], 'item_id': [], 'annotated?': []}

# get all the images in Pathologists folders
# these are copies from those found in the Transplant Raw Slides collection
# the versions in the Pathologist folders should be included, the ones in Transplant Slides Raw should be excluded
pathologists_item_names = []

# looping pathologist folder items
items = get_recursive_items(gc, cf.DSA_PARENT_IDS[0], parent_type='collection')
for item in tqdm(items, total=len(items)):
    pathologists_item_names.append(item['name'])

    data['name'].append(item['name'])
    data['item_id'].append(item['_id'])

    # check if this image has glomeruli annotations
    annotations, _ = annotation_info(gc, item['_id'])

    glom_flag = False

    for annotation in annotations:
        for group in annotation['groups']:
            if group in cf.GLOM_GROUPS:
                glom_flag = True
                break

    if glom_flag:
        data['annotated?'].append('yes')
    else:
        data['annotated?'].append('no')

# looping through transplant slides raw items
items = get_recursive_items(gc, cf.DSA_PARENT_IDS[1], parent_type='collection')
for item in tqdm(items, total=len(items)):
    name = item['name']
    meta = item['meta']

    # PAS images only
    if 'stain' in meta:
        stain = item['meta']['stain']

        if stain == 'PAS' and name not in pathologists_item_names:
            data['name'].append(name)
            data['item_id'].append(item['_id'])

            # check if this image has glomeruli annotations
            annotations, _ = annotation_info(gc, item['_id'])

            glom_flag = False

            for annotation in annotations:
                for group in annotation['groups']:
                    if group in cf.GLOM_GROUPS:
                        glom_flag = True
                        break

            if glom_flag:
                data['annotated?'].append('yes')
            else:
                data['annotated?'].append('no')

# read the data into a DataFrame
df = DataFrame(data)

# organize the dataframe by the annotated? column
# note - you might want to add additional shuffle on the images before assigning train/val/test
df = df.sort_values(by='annotated?', ascending=False).reset_index(drop=True)

# loop through the items to add
df_annotated = df[df['annotated?'] == 'yes']
n = df_annotated.shape[0]  # total images
n_train = int(n * cf.TRAIN_FRACTION)
n_val = n_train + int(n * cf.VAL_FRACTION)

# create the dataset column
dataset_col = ['Not annotated'] * df.shape[0]
for i in range(0, n_train):
    dataset_col[i] = 'Train'
for i in range(n_train, n_val):
    dataset_col[i] = 'Val'
for i in range(n_val, n):
    dataset_col[i] = 'Test'

df['dataset'] = dataset_col

# save to file
df.to_csv(join(datafiles_dir, 'data.csv'), index=False)
