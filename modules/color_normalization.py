from .girder_utils import get_item_image
from histomicstk.saliency.tissue_detection import get_tissue_mask
from histomicstk.preprocessing.color_conversion import rgb_to_lab
import numpy as np


def reinhard_color_stats(gc, item_id, magnification=1.25):
    """Calculate the Reinhard color stats (mean and standard dev. of each channel in LAB color space) for a DSA image
    item. The color stats are calculated from only the pixels that fall within the tissue, as detected by the
    HistomicsTK function: saliency.tissue_detection.get_tissue_mask(..) with default parameters.

    Parameters
    ----------
    gc : girder_client.GirderClient
        authenticated girder client for private images
    item_id : str
        image item id
    magnification : float (optional)
        magnification of thumbnail used to calculate the color stats

    Returns
    -------
    mu : np.array
        LAB mean for each channel (length of 3)
    sigma : np.array
        LAB standard dev. for each channel (length of 3)

    """
    im_info = gc.get('item/{}/tiles'.format(item_id))

    # get thumbnail as specified magnification
    thumbnail = get_item_image(gc, item_id, 'thumbnail', return_type='Array',
                               width=int(im_info['sizeX']*magnification/im_info['magnification']))

    # get the tissue mask
    tissue_mask = get_tissue_mask(thumbnail)[0] == 0

    # convert image to LAB color space
    im_lab = rgb_to_lab(thumbnail)

    # get the pixels inside mask
    tissue_mask_reshaped = tissue_mask[..., None]
    im_lab = np.ma.masked_array(im_lab, mask=np.tile(tissue_mask_reshaped, (1, 1, 3)))

    # calculate the channel's mean and standard deviation
    mu = [im_lab[..., i].mean() for i in range(3)]
    sigma = [im_lab[..., i].std() for i in range(3)]
    return mu, sigma
