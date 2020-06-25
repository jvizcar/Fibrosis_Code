import numpy as np
from math import sqrt
from datetime import datetime


def convert_seconds(seconds):
    """Convert seconds to corresponding hour, minutes and seconds.
    """
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return hour, minutes, seconds


def polygon_area(vertices):
    """Given a list of tuples / list containing the x, y coordinates for a polygon, calculate the area
    enclosed by the polygon via implementation of the Shoelace formula.

    Parameters
    ----------
    vertices : list of tuples
        [(x1, y1), (x2, y2), ... , (xn, yn)], vertices describing the polygon

    Return
    ------
    poly_area : float
        the area enclosed by the polygon

    Sources:
    - https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    - https://en.wikipedia.org/wiki/Shoelace_formula

    """
    # convert list of tuples to numpy arrays
    np_vertices = np.array(vertices)
    xs, ys = np_vertices[:, 0], np_vertices[:, 1]

    # implement Shoelace algorithm
    poly_area = 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    return poly_area


def get_euclidean(point1, point2):
    distance = sqrt(sum([(x - y) ** 2 for x, y in zip(point1, point2)]))
    return distance


def days_between(d1, d2):
    """Find the difference between two dates.

    Source: https://stackoverflow.com/questions/8419564/difference-between-two-dates-in-python

    Parameters
    ----------
    d1 : str
        of form YYYY-MM-DD
    d2 : str
        of form YYYY-MM-DD

    Return
    ------
    d_diff : int
        days between the two dates

    """
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    d_diff = abs((d2 - d1).days)
    return d_diff
