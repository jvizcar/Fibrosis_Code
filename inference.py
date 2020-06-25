"""
Code for pushing the results of DL model as HistomicsUI annotations.
Results are read in from a report csv file created with the predict method of the train model class.
"""
from modules.girder_utils import login
from pandas import read_csv, DataFrame
from os.path import join, splitext
from re import compile
from histomicstk.annotations_and_masks.masks_to_annotations_handler import get_annotation_documents_from_contours
import argparse
from json import load

if __name__ == '__main__':
    # parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_dir", required=True, type=str, help="directory of image dataset")
    ap.add_argument("-v", "--version", required=True, type=str, help="version of model weights to use")
    ap.add_argument("-m", "--mag_factor", required=False, type=int, default=4,
                    help="factor to scale predicted images by, should be 4 if used 10X in trained model for 40X WSI")
    args = vars(ap.parse_args())

    data_dir = args['dataset_dir']
    mag_factor = args['mag_factor']
    model_version = args['version']
    group_names = ['background', 'Glomeruli_Human']
    annprops = {'X_OFFSET': 0, 'Y_OFFSET': 0, 'opacity': 0.2, 'lineWidth': 4.0}
    col_names = ['group', 'color', 'ymin', 'ymax', 'xmin', 'xmax', 'has_holes', 'touches_edge-top', 'touches_edge-left',
                 'touches_edge-bottom', 'touches_edge-right', 'coords_x', 'coords_y']

    # specify the size of images used, scale them appropriately
    im_size = [224, 224]
    im_size = [s * mag_factor for s in im_size]

    # authenticate client
    gc = login(dsa='Transplant')

    # load case information - convert to dict format with filename (no extension) as keys
    cases_info = read_csv('CSVs/images.csv')
    cases_info = cases_info.set_index('name').T.to_dict('dict')

    # read the image report
    test_report = read_csv(join(data_dir, 'weights_v{}_Test_image_report.csv'.format(model_version)))

    # regular expression pattern to extract prediction filename and x, y coordinates
    PATTERN = compile('(?P<filename>.*)_x_(?P<x>\d{1,6})_y_(?P<y>\d{1,6})')

    # annotation documents will be prepended with dataset name and weight version

    dataset_name = data_dir.split('/')[-1]
    DOCNAME_PREFIX = '{}_weights_v{}'.format(dataset_name, model_version)

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
    for item_id, contour_rows in prediction_data.items():
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
