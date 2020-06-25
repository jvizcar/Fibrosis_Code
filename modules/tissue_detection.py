import cv2 as cv
from .girder_utils import get_region_im
import numpy as np
from histomicstk.saliency.tissue_detection import get_tissue_mask


def get_tissue_contours(gc, item_id, magnification=1.25, contour_area_threshold=15000):
    """Get the contours of the tissue in a WSI using lower magnification version of the image. Threshold can be provided
    to remove any small contours.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated client for private images
    item_id : str
        DSA item id of image
    magnification : float (optional)
        magnification to pull image at
    contour_area_threshold : int (optional)
        contours with area smaller than this value will be excluded

    Returns
    -------
    tissue_contours : list
        opencv formatted contours of the tissue
    im : np.ndarray
        RGB low magnification image
    contour_im : np.ndarray
        RGB low magnification image with contours drawn in red

    """
    # get whole image at specific magnification
    im = get_region_im(gc, item_id, {'magnification': magnification})[:, :, :3]

    # get tissue mask using histomicsTK method
    mask = get_tissue_mask(im)[0].astype(np.uint8)
    mask[mask > 0] = 255

    # extract contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)

    # remove any small contours
    tissue_contours = []
    for i, contour in enumerate(contours):
        contour_area = cv.contourArea(contour)
        if contour_area > contour_area_threshold:
            tissue_contours.append(contour)

    # draw the contours on an image copy
    contour_im = cv.drawContours(im.copy(), tissue_contours, -1, [255, 0, 0], 2)
    return tissue_contours, im, contour_im
