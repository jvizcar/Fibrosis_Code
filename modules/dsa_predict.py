from modules.girder_utils import get_region_im

import numpy as np
from pandas import DataFrame

from warnings import catch_warnings, filterwarnings
with catch_warnings():
    filterwarnings("ignore")
    from histomicstk.saliency.tissue_detection import get_tissue_mask
    from histomicstk.annotations_and_masks.masks_to_annotations_handler import get_annotation_documents_from_contours

ANNPROPS = {'X_OFFSET': 0, 'Y_OFFSET': 0, 'opacity': 0.2, 'lineWidth': 4.0}
COL_NAMES = ['group', 'color', 'ymin', 'ymax', 'xmin', 'xmax', 'has_holes', 'touches_edge-top', 'touches_edge-left',
             'touches_edge-bottom', 'touches_edge-right', 'coords_x', 'coords_y']


def dsa_predict(model, gc, item_id, group_name='Positive', ann_doc_name='Default', preprocess_input=None,
                tile_size=(224, 224), save_mag=10, mask_mag=1.25, tissue_threshold=0.3, batch_size=8,
                pred_threshold=0.5, color='rgb(255,153,0)'):
    """Predict on DSA image item, using a grid tiling approach given a binary trained model.
    Parameters
    ----------
    model : tensorflow.keras.models.Model
        a trained keras model for binary classification
    gc : girder_client.GirderClient
        authenticated client, used to get the images
    item_id : str
        image item id
    group_name : str (optional)
        name of the positive class, will be used as the group name in annotation elements
    ann_doc_name : str (optional)
        prepend name of the annotation documents
    preprocess_input : function (optional)
        a function that is applied to the images to process them, works on a tensor-style image
    tile_size : tuple (optional)
        size to predict images at
    save_mag : float (optional)
        magnification to extract tiles at
    mask_mag : float (optional)
        tissue mask is used to decide which tiles to predict on, this is the magnification of the tissue mask
    tissue_threshold : float (optional)
        fraction of tile that must contain tissue to be predicted on
    batch_size : int (optional)
        predictions are done on images in batches
    pred_threshold : float (optinal)
        model predicts a probability from 0 to 1, predictions above pred_threshold are considered the positive class
        that will be pushed as annotations
    color : str (optional)
        rgb(###,###,###) color of element box in annotation element
    Return
    ------
    annotation_data : dict
        annotation data that was pushed as annotation
    """
    # info about the source image
    im_info = gc.get('item/{}/tiles'.format(item_id))
    fr_mag = im_info['magnification']
    fr_width = im_info['sizeX']
    fr_height = im_info['sizeY']

    if save_mag is None:
        # save magnification will be native magnification
        save_mag = fr_mag

    fr_to_lr_factor = save_mag / fr_mag
    # tile size is determined by the save res
    fr_tile_size = int(tile_size[0] / fr_to_lr_factor), int(tile_size[1] / fr_to_lr_factor)  # (width, height)

    # get tissue mask
    lr_im = get_region_im(gc, item_id, {'magnification': mask_mag})[:, :, :3]
    tissue_mask = get_tissue_mask(lr_im)[0]

    # we will loop through image in batches, get the coordinates for batches
    coords = []
    for x in range(0, fr_width, fr_tile_size[0]):
        for y in range(0, fr_height, fr_tile_size[1]):
            # check that the tile won't go over the edge of image, if so skip
            if x + fr_tile_size[0] > fr_width or y + fr_tile_size[1] > fr_height:
                continue

            # check tile for tissue, using the binary mask for tissue
            tissue_tile = tissue_mask[
                          int(y * mask_mag / fr_mag):int((y + fr_tile_size[1]) * mask_mag / fr_mag),
                          int(x * mask_mag / fr_mag):int((x + fr_tile_size[0]) * mask_mag / fr_mag)
                          ]

            # skip if tile does not contain enough tissue
            if np.count_nonzero(tissue_tile) / tissue_tile.size < tissue_threshold:
                continue
            coords.append((x, y))

    # break the coords in batch size chunks
    coord_batches = [coords[i:i + batch_size] for i in range(0, len(coords), batch_size)]
    annotation_data = {col_name: [] for col_name in COL_NAMES}

    print('predicting in batches')
    print('*********************')
    for batch_num, coord_batch in enumerate(coord_batches):
        print('{} of {}'.format(batch_num + 1, len(coord_batches)))
        # get all the images in this batch
        batch_ims = []
        for coord in coord_batch:
            region = {'left': coord[0], 'top': coord[1], 'width': fr_tile_size[0], 'height': fr_tile_size[1],
                      'magnification': save_mag}
            batch_ims.append(get_region_im(gc, item_id, region)[:, :, :3])

        # convert to tensor shape
        batch_ims = np.array(batch_ims)

        # process the image before prediction on it
        batch_ims = preprocess_input(batch_ims) / 255.

        # predict on the batch
        predictions = model.predict(batch_ims)

        # identify predictions that are glomeruli
        for i, pred in enumerate(predictions):
            if pred[0] > pred_threshold:
                # add the data to annotation data
                annotation_data['group'].append(group_name)
                annotation_data['color'].append(color)
                annotation_data['has_holes'].append(0.0)
                annotation_data['touches_edge-top'].append(0.0)
                annotation_data['touches_edge-left'].append(0.0)
                annotation_data['touches_edge-bottom'].append(0.0)
                annotation_data['touches_edge-right'].append(0.0)
                xmin, ymin = coord_batch[i][0], coord_batch[i][1]
                annotation_data['xmin'].append(xmin)
                annotation_data['ymin'].append(ymin)
                xmax = xmin + fr_tile_size[0]
                ymax = ymin + fr_tile_size[1]
                annotation_data['xmax'].append(xmax)
                annotation_data['ymax'].append(ymax)
                annotation_data['coords_x'].append('{},{},{},{}'.format(xmin, xmax, xmax, xmin))
                annotation_data['coords_y'].append('{},{},{},{}'.format(ymin, ymin, ymax, ymax))

    # only push if annotation data is not empty
    n = len(annotation_data['group'])
    if n:
        print('number of tiles to push: {}'.format(n))
        contours_df = DataFrame(annotation_data)
        annotation_docs = get_annotation_documents_from_contours(
            contours_df.copy(), separate_docs_by_group=False, annots_per_doc=100, docnamePrefix=ann_doc_name,
            annprops=ANNPROPS, verbose=False, monitorPrefix=''
        )

        # get current annotations documents from item
        existing_annotations = gc.get('/annotation/item/' + item_id)

        # delete annotation documents starting with the same prefix as about to be pushed
        for ann in existing_annotations:
            if 'name' in ann['annotation']:
                doc_name = ann['annotation']['name']
                if doc_name.startswith(ann_doc_name):
                    gc.delete('/annotation/%s' % ann['_id'])

        # post the annotation documents you created
        for annotation_doc in annotation_docs:
            _ = gc.post(
                "/annotation?itemId=" + item_id, json=annotation_doc)
    else:
        print('no positive tiles to push..')
    return annotation_data