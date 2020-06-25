# Fibrosis Quantification
Updated on 2020-06-24

Code description:
* positive pixel count on WSIs with DSA instance compatability (i.e. pushing PPC results as metadata, 
getting region of interest (ROI) from HistomicsUI annotations, etc.).
* creating datasets from DSA image that has annotations
* training Tensorflow / Keras deep learning models for binary classification problem
    * integration with pushing results of model validation / testing as HistomicsUI annotations for viewing
* using HistomicsUI annotations to automatically create ROIs (developed for renal biopsy images) and push them as
HistomicsUI annotations  

## Requirements
The code was developed and tested using Ubuntu 18.04 LTS and Python version 3.7. 

For DL aspect of project we used Tensorflow 2.0 and Keras, which uses a GPU to speed up the process. The GPU used during
 this project is an NVIDIA GeForce GTX 1060.

Other key Python packages uses: pyvips, histomicstk, girder_client, opencv, scikit-learn, Pillow.

We recommend the code is run using a conda environment. A yaml file is provided to create the conda environment. Two
additional packages need to be installed which are not conda installable: histomicstk and girder_client. To create the
environment run the following commands in the terminal.

```angular2html
$ conda env create -f fibrosis_quant_env.yml
$ conda activate fibrosis_quant_env
$ pip install girder_client
$ pip install histomicstk
```

## Positive Pixel Count (PPC)

Look in modules.positive_pixel_count.count_image(..). Since this version is specific to work with DSA, you must provide
an item_id. Some modification of the code will be needed if you absoultely don't want to use DSA integration. Note that
unlike HistomcisTK version, this version does not use Dask.

Functionality:
* get image tiles from local image filei or DSA image
* run PPC only on ROI which is obtained from HistomicsUI annotations (specify annotation doc and / or annotation group)
* push PPC results as metadata
* create color-coded PPC output images, saved as .npy files. Only create if you are planning to create pyramidal tiffs 
representing the PPC output

## Create Pyramidal Images from .npy Files

Look in modules.pyvips_utils.create_tiff(..). The case_dir parameter should be the one created when running PPC with
save_dir provided. Note that you can specify is_test parameter to True to allow all tiles to be saved, otherwise the 
number of tiles saved based on how many tiles have the given annotation group.

## Parse DSA image with annotations to local image directory

Look in modules.grid_tiling.grid_tiling(..)

Convert a DSA image item (WSI) into a set of local image tiles grouped by their classes. An annotation group or groups
is provided to determine tile class. Note that you can specify is_test parameter to True to allow all tiles to be saved,
otherwise the number of tiles saved is determined on how many tiles have the given annotation group.

Also allows passing in reinhard_stats to create color normalized versions of the images (color augmentation).

## Train and predict

Example on how to use the vgg19 function to train and get result metrics on the train and validation datasets.

```angular2
from modules.keras2 import vgg19

data_dir = 'dir with Train and Val subdirectories'
model = vgg19(data_dir, class_name=['background', 'Glomeruli_Human'])
model.predict(dataset='Train')
model.predict(dataset='Val')
model.predict(dataset='Test')
```

## DSA inference

Push tile predictions as DSA annotations for viewing.

There is a command line script to run this part of the code (inference.py).

```angular2
python inference.py -d dir_with_images_and_model_results -v number_of_version_run_normally_0 -m scale_factor
```

Look at the script for further detail on the command line arguments. Note that some variables in that script are 
specific for the fibrosis example.

## ROI creation

From HistomcisUI annotations, infer the ROI and push it as HistomicsUI annotation.

Below is a small example on how to run this for a single image.

```angular2
from modules.tissue_detection import get_tissue_contours
from modules.annotation_utils import get_element_centers, scale_contours, push_annotations_as_doc
from modules.automatic_roi_algorithm import estimate_ppc_roi

# set the scale factor, which is the magnification to extract tissue contours at over the base magnification
base_mag = gc.get('item/{}/tiles'.format(item_id))['magnification']
scale_factor = MAG / base_mag

# get the tissue contours
tissue_contours, im, _ = get_tissue_contours(gc, item_id, magnification=MAG, contour_area_threshold=500)

# get glomeruli centers at magnification of tissue_contours
glomeruli_centers = get_element_centers(gc, item_id, annotation_doc_names=['doc_name or set to None'],
                                        group_names=['annotation group names'], scale_factor=scale_factor)

if len(glomeruli_centers) > 1:
    # get the roi contours
    roi_contours = estimate_ppc_roi(im, tissue_contours, glomeruli_centers, show=False)

    # scale roi contours to full res
    roi_contours = scale_contours(roi_contours, 1/scale_factor)

    if len(roi_contours):
        # push the roi contours as annotations
        _ = push_annotations_as_doc(gc, item_id, roi_contours, doc_name='Automatic_ROI',
                                    group_name='automatic_roi')
```








