"""
Code for pushing the results of DL model as HistomicsUI annotations.
Results are read in from a report csv file created with the predict method of the train model class.
"""
from os.path import join, splitext
from os import getcwd, listdir

from modules.girder_utils import login
from pandas import read_csv, DataFrame
from re import compile
import configs as cf
from tqdm import tqdm
from histomicstk.annotations_and_masks.masks_to_annotations_handler import get_annotation_documents_from_contours

data_dir = join(getcwd(), 'Images')
mag_factor = 40 // cf.MAG  # this assumes the images are 40x, a smarter solution would be to calculated from # cf.MAG
# and the native magnification of each image

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

group_names = cf.CLASS_NAMES
annprops = {'X_OFFSET': 0, 'Y_OFFSET': 0, 'opacity': 0.2, 'lineWidth': 4.0}
col_names = ['group', 'color', 'ymin', 'ymax', 'xmin', 'xmax', 'has_holes', 'touches_edge-top', 'touches_edge-left',
             'touches_edge-bottom', 'touches_edge-right', 'coords_x', 'coords_y']

# specify the size of images used, scale them appropriately
im_size = cf.TILE_SIZE
im_size = [s * mag_factor for s in im_size]

# authenticate client
gc = login(dsa='Transplant')

# load case information - convert to dict format with filename (no extension) as keys
cases_info = read_csv(join(getcwd(), 'DataFiles/data.csv'))
cases_info = cases_info.set_index('name').T.to_dict('dict')

# read the image report
test_report = read_csv(join(data_dir, f'{model_version.split(".")[0]}_Test_image_report.csv'))

# regular expression pattern to extract prediction filename and x, y coordinates
PATTERN = compile('(?P<filename>.*)_x_(?P<x>\d{1,6})_y_(?P<y>\d{1,6})')

# annotation documents will be prepended with dataset name and weight version
DOCNAME_PREFIX = f'{cf.SAVE_NAME}_{model_version.split(".")[0]}'

# for each filename, create a dictionary to be used in annotation document creation
prediction_data = {}

for i, r in test_report.iterrows():
    # get info from filepath
    filename = splitext(r['filepath'].split('/')[-1])[0]
    m = PATTERN.search(filename).groupdict()

    # get item id from filename
    case_info = cases_info[m['filename'] + '.svs']
    item_id = case_info['item_id']

    # seed the rows for this item if not present
    if item_id not in prediction_data:
        prediction_data[item_id] = {col_name: [] for col_name in col_names}

    true_label = r['true_label']
    pred_label = r['predicted_label']

    # include predictions that were misclassified and correct glomeruli predictions
    if true_label != pred_label:
        if true_label == group_names[1]:
            # misclassified positive class (glomeruli)
            prediction_data[item_id]['group'].append('misclassified_glomeruli')
            prediction_data[item_id]['color'].append('rgb(255,0,0)')
        else:
            # misclassified non-glomeruli
            prediction_data[item_id]['group'].append('misclassified_nonglomeruli')
            prediction_data[item_id]['color'].append('rgb(143,16,222)')
    elif true_label == group_names[1]:
        # case of correctly classified positive class
        prediction_data[item_id]['group'].append('correct_glomeruli')
        prediction_data[item_id]['color'].append('rgb(18,176,60)')
    else:
        # skip this predicion
        continue

    prediction_data[item_id]['ymin'].append(float(m['y']))
    prediction_data[item_id]['ymax'].append(float(m['y']) + im_size[1])
    prediction_data[item_id]['xmin'].append(float(m['x']))
    prediction_data[item_id]['xmax'].append(float(m['x']) + im_size[0])
    prediction_data[item_id]['coords_x'].append(
        '{},{},{},{}'.format(int(m['x']), int(m['x']) + im_size[0], int(m['x']) + im_size[0], int(m['x']))
    )
    prediction_data[item_id]['coords_y'].append(
        '{},{},{},{}'.format(int(m['y']), int(m['y']), int(m['y']) + im_size[1], int(m['y']) + im_size[1])
    )

    prediction_data[item_id]['has_holes'].append(0.0)
    prediction_data[item_id]['touches_edge-top'].append(0.0)
    prediction_data[item_id]['touches_edge-left'].append(0.0)
    prediction_data[item_id]['touches_edge-bottom'].append(0.0)
    prediction_data[item_id]['touches_edge-right'].append(0.0)

# loop through each item id and push to annotations
for item_id, contour_rows in tqdm(prediction_data.items(), total=len(prediction_data)):
    contours_df = DataFrame(contour_rows)
    annotation_docs = get_annotation_documents_from_contours(
        contours_df.copy(), separate_docs_by_group=False, annots_per_doc=100, docnamePrefix=DOCNAME_PREFIX,
        annprops=annprops, verbose=False, monitorPrefix=''
    )

    # get current annotations documents from item
    existing_annotations = gc.get('/annotation/item/' + item_id)

    # delete annotation documents starting with the same prefix as about to be pushed
    for ann in existing_annotations:
        if 'name' in ann['annotation']:
            doc_name = ann['annotation']['name']
            if doc_name.startswith(DOCNAME_PREFIX):
                gc.delete('/annotation/%s' % ann['_id'])

    # post the annotation documents you created
    for annotation_doc in annotation_docs:
        resp = gc.post(
            "/annotation?itemId=" + item_id, json=annotation_doc)
