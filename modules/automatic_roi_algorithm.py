from .misc import get_euclidean

from itertools import combinations
import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize
import itertools
from PIL import Image, ImageDraw

from scipy.sparse.dok import dok_matrix
from scipy.sparse.csgraph import dijkstra

import matplotlib.pyplot as plt


def estimate_ppc_roi(im, tissue_contours, glomeruli_centers, show=False):
    """Draw the ROI on a low magnification image of WSI given contours of the tissue and glomeruli centers in the image.
    Makes use of the Dijkstra approach and skeletonize on the tissue centers to draw the glomeruli. Note that for this
    method it is more beneficial to use lower magnification, such as 0.25, instead of the standard low magnification of
    range 1.25.

    Source: https://stackoverflow.com/questions/43698577/calculating-the-shortest-path-between-two-points-in-a-bitmap-in
    -python

    Parameters
    ----------
    im : np.ndarray
        RGB image of tissue at low resolution
    tissue_contours : list
        opencv style contours of the tissue in the image
    glomeruli_centers : list
        list of glemeruli (x, y) centers
    show : bool (optional)
        set to True to plot some of the results

    Return
    ------
    roi_contours : list
        opencv style contours of the ROI in the image

    """
    roi_contours = []

    # run through each individual tissue
    for tissue_contour in tissue_contours:
        # draw the tissue contour
        tissue_mask = cv.drawContours(np.zeros(im.shape[:-1]), [tissue_contour], -1, 1, cv.FILLED)

        # blur the image but force values between 0 and 1
        tissue_mask = cv.GaussianBlur(tissue_mask, (5, 5), 0)
        tissue_mask = (tissue_mask > 0.).astype(np.uint8)

        # find the glomeruli centers that fall within this tissue contour
        tissue_glom_centers = []
        for center in glomeruli_centers:
            if tissue_mask[center[1], center[0]]:
                tissue_glom_centers.append(center)

        # get the skeleton of the mask
        skeleton = skeletonize(tissue_mask).astype(np.uint8)

        # get the x, y coordinates of the skeleton
        rows, cols = np.where(skeleton)

        # find the closest skeleton point for each glomeruli in tissue
        closest_points = []
        for center in tissue_glom_centers:
            distances = []
            for x, y in zip(cols, rows):
                distances.append(get_euclidean(center, (x, y)))
            # find index of smallest distance
            i = distances.index(min(distances))

            # add the closest point as (x, y)
            closest_points.append([cols[i], rows[i]])

        # need at least 2 glomeruli to draw the roi in a tissue
        if len(closest_points) < 2:
            continue

        # Converting skeleton mask to graph problem to apply Dijkstra method to find shortest path
        def to_index(y, x):
            # translation from 2 coordinates to a single number
            return y * skeleton.shape[1] + x

        def to_coordinates(index):
            # define the reversed translation from index to 2 coordinates
            return index / skeleton.shape[1], index % skeleton.shape[1]

        # build sparse adjacency matrix - two pixels are adjacent in the graph if both are painted
        adjacency = dok_matrix((skeleton.shape[0] * skeleton.shape[1], skeleton.shape[0] * skeleton.shape[1]),
                               dtype=bool)

        # the following lines fills the adjacency matrix by
        directions = list(itertools.product([0, 1, -1], [0, 1, -1]))
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if not skeleton[i, j]:
                    continue

                for y_diff, x_diff in directions:
                    if skeleton[i + y_diff, j + x_diff]:
                        adjacency[to_index(i, j),
                                  to_index(i + y_diff, j + x_diff)] = True

        # convert all the closest points (x, y) to single value, these are known as sources
        sources = [to_index(source[1], source[0]) for source in closest_points]

        # calculate the distant matrix from each source to all possible values in image
        dist_matrix, predecessors = dijkstra(adjacency, directed=False, indices=sources, unweighted=True,
                                             return_predecessors=True)

        # find the two pairs of sources that are farthest away from each other
        combination = list(combinations(range(len(closest_points)), 2))
        distances = []
        for c in combination:
            distances.append(dist_matrix[c[0], sources[c[1]]])

        # find the index with largest value, these indices belong to the sources
        max_combination = combination[distances.index(max(distances))]

        # constructs the path between source and target (the pair of sources that are farthest away from each other)
        source = sources[max_combination[0]]
        target = sources[max_combination[1]]
        pixel_index = target
        pixels_path = []
        while pixel_index != source:
            pixels_path.append(pixel_index)
            pixel_index = predecessors[max_combination[0], pixel_index]

        # create a blank mask to draw only the part of the skeleton connecting the source and target
        roi_mask = Image.new('L', (im.shape[1], im.shape[0]))

        skeleton_points = []
        for pixel_index in pixels_path:
            i, j = to_coordinates(pixel_index)
            skeleton_points.append((int(j), int(i)))
            # im[int(i), int(j)] = [255, 0, 0]

        # use pillow to draw the line with width
        draw = ImageDraw.ImageDraw(roi_mask)
        draw.line(skeleton_points, fill=255, width=40, joint='curve')
        roi_mask = np.array(roi_mask)

        # the width might be too large so do a bit and operation with the tissue mask to remove edges
        roi_mask = cv.bitwise_and(roi_mask, tissue_mask)

        # extract the roi contours
        roi_contour, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)

        # if the points are too close together at low resolution, then there will be no contours to draw, skip these
        if len(roi_contour) > 0:
            # append the first contours
            roi_contours.append(roi_contour[0])

    if show:
        tissue_mask = cv.drawContours(np.zeros(im.shape[:-1]), tissue_contours, -1, 1, cv.FILLED)
        roi_mask = cv.drawContours(np.zeros(im.shape[:-1]), roi_contours, -1, 1, cv.FILLED)
        im_with_roi = cv.drawContours(im.copy(), roi_contours, -1, [255, 0, 0], 2)

        # plot the original image, tissue mask, and roi_mask, draw the roi contous on original image
        fig, ax = plt.subplots(ncols=3, figsize=(10, 5))
        ax[0].imshow(im_with_roi)
        ax[0].set_title('Image with ROI contours', fontsize=14)
        ax[1].imshow(tissue_mask)
        ax[1].set_title('Tissue Mask', fontsize=14)
        ax[2].imshow(roi_mask)
        ax[2].set_title('ROI Mask', fontsize=14)
        plt.show()

    return roi_contours