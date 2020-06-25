import os
import pyvips
import re
import numpy as np
from os.path import join
from os import remove

# regex pattern to extract row ind and col ind from tile file names
PATTERN = re.compile("(?P<filename>.*)_left_(?P<left>\d{1,6})_top_(?P<top>\d{1,6})-rowInd_(?P<row>\d{1,3})_colInd_(?P<col>\d{1,3}).*")

# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

# numpy array to vips image
def numpy2vips(a):
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi


# vips image to numpy array
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


def create_tiff(tile_dir, cleanup=False):
    """Build a mosaic image (tile-based image) from a directory of tile images saved as npy files. These files are
     generated jc_toolkits.quantification.positive_pixel_count.count_image().
    The column and row index are in the filename for creating the mosaic.
    Mosaic image is saved in tile_dir using the filename.svs name.

    # source: https://libvips.github.io/libvips/API/current/Examples.md.html
    # source 2: https://libvips.github.io/libvips/API/current/Examples.md.html
    # source 3: https://github.com/libvips/pyvips/issues/109
    # source 4: https://github.com/libvips/libvips/issues/1254
    Mohamed Amgad provided the source code, I added some modification.

    Parameters
    ----------
    tile_dir: str
        directory containing the tile images, name convention: filename-rowInd_#_colInd_#.npy
    cleanup : bool (optional)
        if True then the npy files are removed after tiff creation

    Return
    ------
    save_path : str
        the path to where the image was saved
    filename : str
        the filename wihtout the left, top part
    left : int
        left coordinate of where output image starts
    top : int
        top coordinate of where output image starts

    """
    # sort the tile images into a dict with keys being row and column location
    tile_paths = {}
    for j in os.listdir(tile_dir):
        if j.endswith('.npy'):
            m = PATTERN.search(j).groupdict()
            row_ind, col_ind = int(m['row']), int(m['col'])
            if row_ind not in tile_paths:
                tile_paths[row_ind] = {}

            tile_paths[row_ind][col_ind] = os.path.join(tile_dir, j)

    # this makes a 8-bit, mono image (initializes as 1x1x3 matrix)
    im = pyvips.Image.black(1, 1, bands=3)

    # build image by rows
    for r in range(len(tile_paths)):
        row_im = pyvips.Image.black(1, 1, bands=3)

        for c in range(len(tile_paths[0])):
            tilepath = tile_paths[r][c]

            tile = numpy2vips(np.load(tilepath))
            # tile = pyvips.Image.new_from_file(tilepath, access="sequential")
            row_im = row_im.insert(tile, row_im.width, 0, expand=True)

        # insert row
        im = im.insert(row_im, 0, im.height, expand=True)

    # save the pyramidal tiff image
    filename, left, top = m['filename'], int(m['left']), int(m['top'])
    save_path = join(tile_dir, filename + '.tiff')
    im.tiffsave(save_path, tile=True, tile_width=240, tile_height=240, pyramid=True, compression='lzw')

    if cleanup:
        # remove the npy files
        for r in range(len(tile_paths)):
            for c in range(len(tile_paths[0])):
                tilepath = tile_paths[r][c]
                remove(tilepath)
    return save_path, filename, left, top
