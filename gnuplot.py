"""
Export data in text files that can be read by gnuplot
"""
import os
import numpy as np


def export_gnuplot_mat(fname, mat, x=None, y=None):
    """
    Export a matrix to a text file in an appropriate format to be plotted using gnuplots plot matrix command.
    Example gnuplot syntax: "plot fname matrix nonuniform with image"

    In this format the first column gives the y-coordinates of the image, the top row gives the x-coordinates of
    the image, and the remaining entries define the image.

    :param fname: path to save text file
    :param mat: matrix
    :param x: x-coordinates of matrix (i.e. along axis=1)
    :param y: y-coordinates of matrix (i.e. along axis=0)
    :return:
    """

    if x is None or y is None:
        x = np.arange(mat.shape[1])
        y = np.arange(mat.shape[0])

    n = len(x) + 1
    first_row = np.concatenate((np.array([n]), x), axis=0)[None, :]
    other_rows = np.concatenate((y[:, None], mat), axis=1)
    data = np.concatenate((first_row, other_rows), axis=0)
    np.savetxt(fname, data, delimiter='\t')


def export_gnuplot_splot_mat(fname, mat, x=None, y=None, z=None):
    """
    Export a matrix to a text file in an appropriate format to be plotted using gnuplots splot command

    This file is expected to be lines of x, y, z, f(x,y,z) form, where the matrix columns are listed in order for a
    single row. A newline appears between data representing each row. This format is appropriate for data which should
    be displayed like a matrix, but which may not consist of uniformaly spaced points.

    The gnuplot syntax is e.g. "splot fname using 1:2:3:4 with lines".

    :param fname: path to save file
    :param mat: 2D array
    :param x: x-coordinates. Should be array of same dimensions as mat.
    :param y: y-coordinates. Should be array of same dimensions as mat.
    :param z: z-coordinates. Should be array of same dimensions as mat.
    :return:
    """

    ny, nx = mat.shape

    if x is None or y is None:
        x, y = np.meshgrid(range(nx), range(ny))

    if z is None:
        z = np.zeros(mat.shape)

    with open(fname, 'ab') as f:
        # loop over rows.
        for ii in range(nx):
            data = np.concatenate((x[:, ii][:, None], y[:, ii][:, None], z[:, ii][:, None], mat[:, ii][:, None]), axis=1)
            np.savetxt(f, data, delimiter=' ', )

            # newline between rows, but not after last row
            if ii < (nx - 1):
                f.write('\n'.encode('utf-8'))


def export_gnuplot_rgb(fname, mat, x=None, y=None):
    """
    Export rgb data to be plotted in 2D

    The gnuplot syntax is e.g. "plot fname using with rgbimage" or
    "plot fname with 1:2:3:4:5 using rgbimage"

    :param fname:
    :param mat: 3 x ny x nx image
    :param x:
    :param y:

    :return:
    """

    nc, ny, nx = mat.shape

    if nc != 3:
        raise Exception("Matrix was wrong shape. Should be 3 x ny x nx, where first three are RGB")

    if x is None or y is None:
        x, y = np.meshgrid(range(nx), range(ny))

    if os.path.exists(fname):
        raise Exception("Path %s already exists" % fname)

    with open(fname, 'ab') as f:
        # loop over rows.
        for ii in range(nx):
            data = np.concatenate((x[:, ii][:, None], y[:, ii][:, None],
                                   mat[0, :, ii][:, None], mat[1, :, ii][:, None], mat[2, :, ii][:, None]),
                                  axis=1)
            np.savetxt(f, data, delimiter=' ', )

            # newline between rows, but not after last row
            if ii < (nx - 1):
                f.write('\n'.encode('utf-8'))
