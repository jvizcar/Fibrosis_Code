from os.path import splitext, join
from os import makedirs
from warnings import catch_warnings, filterwarnings
from copy import deepcopy
import pandas as pd
import numpy as np
from imageio import imwrite
from random import Random
from .girder_utils import get_region_im
from .color_normalization import reinhard_color_stats

with catch_warnings():
    # currently histomicsTK is giving warnings about depracated sklearn kde module, this catch prevents this warning
    # - from showing up
    filterwarnings("ignore")
    from histomicstk.annotations_and_masks.annotation_and_mask_utils import get_scale_factor_and_appendStr, \
        scale_slide_annotations, get_bboxes_from_slide_annotations
    from histomicstk.annotations_and_masks.annotations_to_masks_handler import get_image_and_mask_from_slide
    from histomicstk.saliency.tissue_detection import get_tissue_mask
    from histomicstk.preprocessing.color_normalization import reinhard

# global variables used in grid_tiling(..)
# - sets up the key-word arguments used in annotation_and_masks HTK module functions
GET_ROI_MASK_KWARGS = {
    'iou_thresh': 0.0,
    'crop_to_roi': True,
    'use_shapely': True,
    'verbose': False
}
GET_CONTOURS_KWARGS = {
    'groups_to_get': None,
    'get_roi_contour': False,
    'discard_nonenclosed_background': True,
    'MIN_SIZE': 0, 'MAX_SIZE': None,
    'verbose': False, 'monitorPrefix': ""
}
GET_KWARGS = {
    'gc': None, 'slide_id': None,
    'GTCodes_dict': None,
    'MPP': None,
    'MAG': None,
    'get_roi_mask_kwargs': GET_ROI_MASK_KWARGS,
    'get_contours_kwargs': GET_CONTOURS_KWARGS,
    'get_rgb': True,
    'get_contours': False,
    'get_visualization': False,
}


def grid_tiling(gc, item_id, group_names, save_dir, save_mag=None, mask_mag=1.25, tile_size=(224, 224),
                tissue_threshold=0.3, annotation_threshold=0.15, random_seed=64, is_test=False,
                oversample_background=2.0, reinhard_stats=None):
    """Split a DSA image item (WSI) into smaller images and save locally grouped by annotations. This approach grids the
    image into equal sized small images, or tiles (i.e. a grid is placed over the WSI starting at the top left corner).
    At the bottom and right edge of the WSI the tiles are ignored if not of correct size (the case where the WSI
    dimensions are not a multiple factor of the tile size). A list of annotation group names are needed to group the
    tiles into classes of images saved in their own directories. Tiles with no tissue detected are ignored and tiles not
    containing annotations (but have tissue) are by default saved into background class. A background annotation group
    will cause issues so avoid having this annotation group name.

    Tiles can be saved at a lower magnification than source image if needed (param: save_mag). Note that tiles size
    specified should be the tile size at the save magnification not the source magnification. Image saved will be of the
    tile size specified in parameters, regardless of the save_mag used.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated client
    item_id : str
        DSA image item id
    group_names : list
        list of annotation group names
    save_dir : str
        directory to create group directories with images - save_dir / group_name_1, save_dir / background, etc.
    save_mag : float (optional)
        magnification to use when saving the images, if None then source magnification will be used.
    mask_mag : float (optional)
        magnification to create binary mask of tissue and annotations. Note that if your annotations are very small
        it will benefit to use a larger value than default here, but binary masks will fail to create at very high
        magnifications.
    tile_size : tuple (optional)
        size (width, height) to save tiles at, note that this is the size it will be saved at regardless of the
        magnification used to save the images at (i.e. if save_mag is 4 times less than the source magnification than
        the actual tile_size will represent 4 times the pixels at full resolution).
    tissue_threshold : float (optional)
        from 0 to 1, percentage of tile that must contain tissue to be included
    annotation_threshold : float (optional)
        from 0 to 1, percentage of tile that must contain annotation (per group) to be labeled as annotation. Note
        that a single tile may be saved twice, representing multiple classes.
    random_seed : int (optional)
        random seed to use when shuffling the background regions
    is_test : bool (optional)
        if True then all the background regions will be saved, otherwise oversample_background will be used to determine
        how many background regions to save
    oversample_background : float (optional)
        factor to oversample background class images, compared to the number of images of the class of annoation images
        with the most images saved
    reinhard_stats : dict (optional)
        if not None then the images saved will be color augmented by color normalizing the tiles using the Reinhard
        color norm method. This dict should contain src_mu and scr_sigma keys with the stats for this image and
        target_mu and targe_sigma keys which are lists contain 1 or more target images to normalize to.

    """
    im_info = gc.get('item/{}/tiles'.format(item_id))
    if save_mag is None:
        save_mag = im_info['magnification']

    if reinhard_stats is not None:
        # get color stats for image
        mu, sigma = reinhard_color_stats(gc, item_id)

    # ----- prep work ----- #
    filename = splitext(gc.getItem(item_id)['name'])[0]

    # create dirs for each image class to save
    group_dirs = [join(save_dir, group_name) for group_name in group_names]
    for group_dir in group_dirs:
        makedirs(group_dir, exist_ok=True)
    background_dir = join(save_dir, 'background')
    makedirs(background_dir, exist_ok=True)

    # get image annotations
    annotations = gc.get('/annotation/item/' + item_id)

    # create a dataframe to use with annotation to mask handler functions (gt codes)
    gt_data = [
        [group_name, 1, i + 1, 0, 0, 'rgb(0, 0, {})'.format(i), ''] for i, group_name in enumerate(group_names)
    ]
    gt_codes = pd.DataFrame(columns=['group', 'overlay_order', 'GT_code', 'is_roi', 'is_background_class', 'color',
                                     'comments'], data=gt_data, index=range(len(group_names)))
    gt_codes.index = gt_codes.loc[:, 'group']

    # get binary masks - tissue mask and annotation(s) mask
    mask_mag_factor, _ = get_scale_factor_and_appendStr(gc=gc, slide_id=item_id, MAG=mask_mag)
    # - scaling the annotations to lower magnification
    mask_annotations = scale_slide_annotations(deepcopy(annotations), sf=mask_mag_factor)

    # - binary masks are for the whole image at low resolution, function returns also the RGB image which we use for
    # - getting the tissue mask
    mask_element_info = get_bboxes_from_slide_annotations(mask_annotations)
    get_kwargs = deepcopy(GET_KWARGS)  # avoid referencing on the global variable
    get_kwargs['gc'] = gc
    get_kwargs['slide_id'] = item_id
    get_kwargs['GTCodes_dict'] = gt_codes.T.to_dict()
    get_kwargs['bounds'] = None
    get_kwargs['MAG'] = mask_mag
    ann_mask_and_image = get_image_and_mask_from_slide(mode='wsi', slide_annotations=mask_annotations,
                                                       element_infos=mask_element_info, **get_kwargs)
    tissue_mask = get_tissue_mask(ann_mask_and_image['rgb'])[0]

    # convert the annotations to lower magnification
    fr_to_lr_factor, _ = get_scale_factor_and_appendStr(gc=gc, slide_id=item_id, MAG=save_mag)
    annotations = scale_slide_annotations(annotations, sf=fr_to_lr_factor)
    lr_element_info = get_bboxes_from_slide_annotations(annotations)

    # get full resolution information for image
    fr_mag = im_info['magnification']
    fr_width = im_info['sizeX']
    fr_height = im_info['sizeY']
    fr_tile_size = int(tile_size[0] / fr_to_lr_factor), int(tile_size[1] / fr_to_lr_factor)  # (width, height)

    # change the get_kwargs to save magnification
    get_kwargs['MAG'] = save_mag

    # ----- loop through image at full res ----- #
    group_annotation_counts = [0] * len(group_names)
    background_regions = []
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

            # check tile for annotations, using the binary mask for annotations
            annotation_tile = ann_mask_and_image['ROI'][
                              int(y * mask_mag / fr_mag):int((y + fr_tile_size[1]) * mask_mag / fr_mag),
                              int(x * mask_mag / fr_mag):int((x + fr_tile_size[0]) * mask_mag / fr_mag)
                              ]

            # tile is background if no annotation is present (of any group)
            background_flag = True
            # - check for each annotation group
            for i, group_name in enumerate(group_names):
                group_annotation_tile = annotation_tile == i + 1

                # tile is ignored if not enough contain annotation
                if np.count_nonzero(group_annotation_tile) / group_annotation_tile.size < annotation_threshold:
                    continue

                background_flag = False
                group_annotation_counts[i] += 1

                # get annotation image and save it
                get_kwargs['bounds'] = {
                    'XMIN': x, 'XMAX': x + fr_tile_size[0],
                    'YMIN': y, 'YMAX': y + fr_tile_size[1]
                }

                annotation_im = get_image_and_mask_from_slide(mode='manual_bounds', slide_annotations=annotations,
                                                              element_infos=lr_element_info, **get_kwargs)['rgb']

                # save the image to correct directory
                imwrite(join(group_dirs[i], '{}_x_{}_y_{}.png'.format(filename, x, y)), annotation_im)

                if reinhard_stats is not None:
                    # add color augmentation with Reinhard method
                    for j, (_, v) in enumerate(reinhard_stats.items()):
                        im_norm = reinhard(annotation_im.copy(), v['mu'], v['sigma'], src_mu=mu, src_sigma=sigma)
                        imwrite(join(group_dirs[i], '{}_x_{}_y_{}_norm_{}.png'.format(filename, x, y, j)), im_norm)

            if background_flag:
                # save coordinates for non-glomeruli images candidates
                background_regions.append({'magnification': save_mag, 'left': x, 'top': y, 'width': fr_tile_size[0],
                                           'height': fr_tile_size[1]})

    # randomly select background class coordinates
    # - oversample the background class by a factor of the most represented annoation class
    Random(random_seed).shuffle(background_regions)
    if not is_test:
        background_regions = background_regions[:int(oversample_background * max(group_annotation_counts))]

    for region in background_regions:
        tile_im = get_region_im(gc, item_id, region)[:, :, :3]

        # save background image
        imwrite(join(background_dir, '{}_x_{}_y_{}.png'.format(filename, region['left'], region['top'])), tile_im)

        if reinhard_stats is not None:
            # add color augmentation with Reinhard method
            for j, (_, v) in enumerate(reinhard_stats.items()):
                im_norm = reinhard(tile_im.copy(), v['mu'], v['sigma'], src_mu=mu, src_sigma=sigma)
                imwrite(join(background_dir, '{}_x_{}_y_{}_norm_{}.png'.format(filename, region['left'],
                                                                               region['top'], j)), im_norm)
