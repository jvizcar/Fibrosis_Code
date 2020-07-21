"""
Use a trained VGG19 model to predict glomeruli directly on the DSA images that have no ground truth. The results are
pushed directly to the DSA as annoations.
"""
from modules.keras2 import vgg19
from modules.girder_utils import login
from modules.dsa_predict import dsa_predict

from os import getcwd, listdir
from os.path import join

from tensorflow.keras.applications.vgg19 import preprocess_input
from pandas import read_csv
from time import time
from re import compile
import configs as cf

# static variables
CLASS_NAMES = ['background', 'glomerulus']
GROUP_NAME = 'predicted_glomeruli'
DATASETS = ['Not annotated']

data_dir = join(getcwd(), 'Images')

# get model version
pattern = compile('weights_v(?P<version>\d{1,3}).hdf5')
if cf.MODEL_VERSION == 'latest':
    files = listdir(data_dir)

    versions = []
    for file in files:
        m = pattern.search(file)

        if m:
            m = m.groupdict()
            versions.append(int(m['version']))

    max_version = max(versions)

    model_version = f'weights_v{max_version}.hdf5'
else:
    model_version = f'weights_v{cf.MODEL_VERSION}.hdf5'

# load model
model = vgg19(data_dir, weights_filename=model_version, class_names=CLASS_NAMES).model
ann_doc_name = f'{cf.SAVE_NAME}_{model_version.split(".")[0]}'

# authenticate client
gc = login(dsa='Transplant')

# read the images info
im_info = read_csv(join(getcwd(), 'DataFiles/data.csv'))

# subset to only the two datasets
im_info = im_info[im_info['dataset'].isin(DATASETS)].reset_index(drop=True)

# time the process
t = time()
for i, r in im_info.iterrows():
    item_id = r['item_id']
    print('image {} of {}'.format(i+1, im_info.shape[0]))

    run_flag = True
    # get annotations...
    for annotation_doc in gc.get(f'annotation/item/{item_id}'):
        annotation = annotation_doc['annotation']

        if 'name' in annotation and annotation['name'].startswith(data_dir):
            run_flag = False

    if run_flag:
        # predict glomeruli on all the hold-out images
        print('predicting on {}'.format(r['name']))
        _ = dsa_predict(model, gc, item_id, group_name=GROUP_NAME, ann_doc_name=ann_doc_name,
                        preprocess_input=preprocess_input)
t = time() - t
print('time taken: {:.0f} hours, {:.0f} min, {:.0f} sec'.format(t//3600, t//60, t % 60))
