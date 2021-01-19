"""
Miscellaneous helper functions. As I collect many relating to a certain area, these can be split into topic
specific modules.
"""

import numpy as np
import scipy.optimize
import scipy.sparse as sp
from scipy import fft
from PIL import Image
import PIL.TiffTags
import tiffile
import os
import copy
import glob
#import libtiff
import datetime
import json
from pathlib import Path
import pandas as pd
import re

# I/O for metadata files
def parse_mm_metadata(metadata_dir, file_pattern="*metadata*.txt"):
    """
    Parse all micromanager metadata files in subdirectories of metadata_dir. MM metadata is stored as JSON
    object in text file.

    :param str metadata_dir: directory storing metadata files
    :param str file_pattern: file pattern which metadata files must match

    :return: df_images: dataframe object summarizing images described in metadata file
    :return: dims:
    :return: summary:
    """

    if not os.path.exists(metadata_dir):
        raise Exception("Path '%s' does not exists." % metadata_dir)

    # todo: are there cases where there are multiple metadata files for one dataset?
    metadata_paths = list(Path(metadata_dir).glob('**/' + file_pattern))
    metadata_paths = sorted(metadata_paths)

    if metadata_paths == []:
        raise Exception("No metadata files matching pattern '%s' found." % file_pattern)

    # open first metadata and get roi_size few important pieces of information
    with open(metadata_paths[0], 'r') as f:
        datastore = json.load(f)

    # get summary data
    summary = datastore['Summary']
    dims = {}
    for k, entry in summary['IntendedDimensions'].items():
        dims[k] = entry

    for k, entry in summary['UserData'].items():
        dims[k] = entry['scalar']

    # run through each metadata file to figure out settings for stage positions and individual images
    initialized = False
    multipage_tiff_style = False
    titles = []
    userdata_titles = []
    extra_titles = []
    data = []
    for filename in metadata_paths:

        with open(filename, 'r') as f:
            datastore = json.load(f)

        for k, entry in datastore.items():

            # skip items we don't care much about yet
            if k == 'Summary':
                continue

            # separate coordinate data stored in single page TIFF files style metadata
            if re.match("Coords-.*", k):
                continue

            # get column titles from metadata
            # get titles
            if not initialized:
                # check for multipage vs single page tiff style
                m = re.match('FrameKey-(\d+)-(\d+)-(\d+)', k)
                if m is not None:
                    multipage_tiff_style = True

                # get titles
                for kk in entry.keys():
                    if kk == 'UserData':
                        for kkk in entry[kk].keys():
                            userdata_titles.append(kkk)
                    else:
                        titles.append(kk)

                if multipage_tiff_style:
                    # these
                    extra_titles = ['Frame', 'FrameIndex', 'PositionIndex', 'Slice', 'SliceIndex', 'ChannelIndex']
                extra_titles += ["directory"]
                initialized = True

            # accumulate data
            data_current = []
            for t in titles:
                data_current.append(entry[t])
            for t in userdata_titles:
                # todo: maybe need to modify this more generally for non-scalar types...
                data_current.append(entry['UserData'][t]['scalar'])

            if multipage_tiff_style:
                # parse FrameKey information
                m = re.match('FrameKey-(\d+)-(\d+)-(\d+)', k)

                time_index = int(m.group(1))
                channel_index = int(m.group(2))
                z_index = int(m.group(3))

                m = re.match('Pos-(\d+)', entry['PositionName'])
                if m is not None:
                    position_index = int(m.group(1))
                else:
                    position_index = 0

                data_current += [time_index, time_index, position_index, z_index, z_index, channel_index]

            # this is also stored in "extra titles"
            data_current += [os.path.dirname(filename)]


            # combine all data
            data.append(data_current)

    # have to do some acrobatics to get slice in file info
    userdata_titles = ['User' + t for t in userdata_titles]
    image_metadata = pd.DataFrame(data, columns=titles+userdata_titles+extra_titles)

    # for TIF files containing multiple images, we need the position in the file for each image
    fnames = image_metadata['FileName'].unique()

    image_pos_in_file = np.zeros((image_metadata.shape[0]), dtype=np.int)

    if multipage_tiff_style:
        for fname in fnames:
            inds = (image_metadata['FileName'] == fname)
            current_pos = image_metadata['ImageNumber'][inds]
            image_pos_in_file[inds] = current_pos - current_pos.min()

    image_metadata['ImageIndexInFile'] = image_pos_in_file

    return image_metadata, dims, summary

def read_dataset(md, time_indices=None, channel_indices=None, z_indices=None, xy_indices=None, user_indices={}, dtype=np.uint16):
    """
    Load a set of images from MicroManager metadata, read using parse_mm_metadata()

    :param md: metadata Pandas datable, as created by parse_mm_metadata()
    :param time_indices:
    :param channel_indices:
    :param z_indices:
    :param xy_indices:
    :param user_indices: {"name": indices}
    :param file_pattern:
    :return:
    """

    # md, dims, summary = parse_mm_metadata(dir)

    def check_array(arr, ls):
        to_use = np.zeros(arr.shape, dtype=np.bool)
        for l in ls:
            to_use = np.logical_or(to_use, arr == l)

        return to_use

    to_use = np.ones(md.shape[0], dtype=np.bool)

    if time_indices is not None:
        if not isinstance(time_indices, list):
            time_indices = [time_indices]
        to_use = np.logical_and(to_use, check_array(md["FrameIndex"], time_indices))

    if xy_indices is not None:
        if not isinstance(xy_indices, list):
            xy_indices = [xy_indices]
        to_use = np.logical_and(to_use, check_array(md["PositionIndex"], xy_indices))

    if z_indices is not None:
        if not isinstance(z_indices, list):
            z_indices = [z_indices]
        to_use = np.logical_and(to_use, check_array(md["SliceIndex"], z_indices))

    if channel_indices is not None:
        if not isinstance(channel_indices, list):
            channel_indices = [channel_indices]
        to_use = np.logical_and(to_use, check_array(md["ChannelIndex"], channel_indices))

    for k, v in user_indices.items():
        if not isinstance(v, list):
            v = [v]
        to_use = np.logical_and(to_use, check_array(md[k], v))

    fnames = [os.path.join(d, p) for d, p in zip(md["directory"][to_use], md["FileName"][to_use])]
    slices = md["ImageIndexInFile"][to_use]
    imgs = load_images(fnames, slices, dtype=dtype)

    return imgs

# I/O for image files
def read_tiff(fname, slices=None, offset=None, skip=None, dtype=np.uint16):
    """
    Read tiff file containing multiple images

    # todo: PIL uses int32 for memory mapping on windows, so cannot address parts of TIF file after 2GB
    # todo: stack overflow https://stackoverflow.com/questions/59485047/importing-large-multilayer-tiff-as-numpy-array
    # todo: pull request https://github.com/python-pillow/Pillow/pull/4310
    # todo: currently PYPI does not have this fix, but Christoph Goehlke version does
    # todo: see here: https://www.lfd.uci.edu/~gohlke/pythonlibs/

    :param fname: path to file
    :param slices: list of slices to read
    :param offset: offset image to start reading. Will be ignored unless slices is None. This is useful if
    the number of slices is unknown.
    :param skip: images to skip between reading. Will be ignored unless slices is None. This is useful if the number
    of slices is unknown.
    :param dtype: NumPy datatype of output array

    :return imgs: 3D numpy array, nz x ny x nx
    :return tiff_metadata: metadata tags with recognized numbers, corresponding to known descriptions
    :return other_metadata: other metadata tags with unrecognized numbers
    """
    if isinstance(slices, int) or np.issubdtype(type(slices), np.integer):
        slices = [slices]

    tif = Image.open(fname)

    # read tiff and micro manager metadata
    tiff_metadata = {}
    other_metadata = {}
    for key in tif.tag_v2.keys():
        try:
            tiff_metadata[PIL.TiffTags.TAGS[key]] = tif.tag[key]
        except KeyError:
            # other_metadata = json.loads(tif.tag[key][0])
            other_metadata.update({str(key): tif.tag[key]})

    n_frames = tif.n_frames

    # try:
    #     ij_data = parse_imagej_tiff_tag(tiff_metadata['ImageDescription'][0])
    #     n_ij = int(ij_data['images'])
    #     if n_frames == 1 and n_ij != 1:
    #         n_frames = n_ij
    # except:
    #     pass

    # tif = libtiff.TIFF.open(path, 'r')
    # surely there is roi_size property to find the number of images? But couldn't find it...
    # n_frames = sum([1 for _ in tif.iter_images()])

    # check arguments are consistent
    if slices is not None and offset is None and skip is None:
        pass
    elif slices is None and offset is None and skip is None:
        slices = range(n_frames)
    elif slices is None and offset is not None and skip is not None:
        slices = range(offset, n_frames, skip)
    else:
        raise Exception('incompatible arguments provided for slices, offsets, and skip. '
                        'Either provide slices and no skip or offset, or provide roi_size skip and offset but no slices.')

    # load images
    imgs = []
    for s in slices:
        tif.seek(s)
        imgs.append(np.asarray(tif, dtype=dtype))

    # for ii, tif_slice in enumerate(tif.iter_images()):
    #     if ii in slices:
    #         imgs.append(np.asarray(tif_slice, dtype=np.uint16))

    return np.asarray(imgs), tiff_metadata, other_metadata

def load_images(fnames, slice_indices, dtype=np.uint16):
    """
    Load multiple images and slices, definted by lists fnames and slice_indices,
    and return in same order as inputs. Automatically load all images from each file without multiple reads.

    # todo: right now only supports tifs, but in general could support other image types
    # todo: should also return metadata

    :param fnames:
    :param slice_indices:

    :return imgs:
    """
    # counter for files
    inds = np.arange(len(fnames))
    slice_indices = np.array(slice_indices)
    # only want to load each tif once, in case roi_size single tif has multiple images
    fnames_unique = list(set(fnames))
    fnames_unique.sort()

    # Tells us which unique filename the slice_indices correspond to
    inds_to_unique = np.array([i for fn in fnames for i, fnu in enumerate(fnames_unique) if fn == fnu])

    # load tif files, store results in roi_size list
    imgs = [''] * len(fnames)
    for ii, fu in enumerate(fnames_unique):
        slices = slice_indices[inds_to_unique == ii]
        imgs_curr, _, _ = read_tiff(fu, slices, dtype=dtype)

        # this is necessary in case e.g. one file has images [1,3,7] and another has [2,6,10]
        for jj, ind in enumerate(inds[inds_to_unique == ii]):
            imgs[ind] = imgs_curr[jj, :, :]

    if any([isinstance(im, str) for im in imgs]):
        raise Exception()
    imgs = np.asarray(imgs)

    return imgs

def save_tiff(img, save_fname, dtype='uint16', tif_metadata=None, other_metadata=None,
              axes_order='ZYX', hyperstack=False, **kwargs):
    """
    Save an nD NumPy array to a tiff file

    # todo: currently not saving metadata

    :param img: nd-numpy array to save as TIF file
    :param save_fname: path to save file
    :param dtype: data type to save tif as
    :param tif_metadata: dictionary of tif metadata. All tags must be recognized.
    :param other_metadata: dictionary of tif metadata with tags that will not be recognized.
    :param axes_order: a string consisting of XYZCTZ
    :param hyperstack: whether or not to save in format compatible with imagej hyperstack
    :return:
    """

    if not isinstance(dtype, str):
        raise Exception("dtype must be a string, e.g. 'uint16'")

    # convert numpy array images to PIL.Image
    # if img.ndim == 2:
    #     im_list = [Image.fromarray(img.astype(dtype))]
    # elif img.ndim == 3:
    #     im_list = [Image.fromarray(img[ii].astype(dtype)) for ii in range(img.shape[0])]
    # elif img.ndim > 3:
    #     n = int(img.size / (img.shape[-2] * img.shape[-1]))
    #     img_reshaped = img.reshape([n, img.shape[-2], img.shape[-1]])
    #     im_list = [Image.fromarray(img_reshaped[ii].astype(dtype)) for ii in range(img_reshaped.shape[0])]
    # else:
    #     raise Exception('number of dimensions was %d, but must be >1.' % img.ndim)

    # coerce metadata # todo: this has not worked so far...
    # ifd = ImageFileDirectory_v2()
    # for k, v in tif_metadata.items():
        # tag_info = PIL.TiffTags.TagInfo(name=k)
        # ifd[k] = v

    # im_list[0].save(save_fname, save_all=True, append_images=im_list[1:], tiffinfo=ifd)
    # im_list[0].save(save_fname, save_all=True, append_images=im_list[1:])

    if hyperstack:
        img = tiffile.transpose_axes(img, axes_order, asaxes='TZCYXS')

    tiffile.imwrite(save_fname, img.astype(dtype), dtype=dtype, imagej=True, **kwargs)

def parse_imagej_tiff_tag(tag):
    """
    Parse information from the TIFF "ImageDescription" tag saved by ImageJ. This tag has the form
    "key0=val0\nkey1=val1\n..."

    These tags can be obtained from the TIFF metadata using the read_tiff() function

    :param tag: String
    :return subtag_dict: dictionary {key0: val0, ..., keyN: valN}
    """
    subtag_dict = {}

    subtags = tag.split("\n")
    for st in subtags:
        try:
            k, v = st.split("=")
            subtag_dict.update({k: v})
        except ValueError:
            # ignore empty tags
            pass

    return subtag_dict

def tiffs2stack(fname_out, dir_path, fname_exp="*.tif", dtype=np.float,
                exp=r"(?P<prefix>.*)nc=(?P<channel>\d+)_nt=(?P<time>\d+)_nxy=(?P<position>\d+)_nz=(?P<slice>\d+)"):
    """
    Combine single TIFF files into a stack based on name

    :param fname_out: file name to save result
    :param dir_path: directory to search for files
    :param fname_exp: only files matching this wildcard expression will be considered, e.g. "widefield*.tif"
    :param exp: regular expression matching the file name (without the directory or file extension). Capturing groups
    must be named "prefix" "channel", "time", "position", and "slice". If one or more of these is absent, results
    will still be correct

    :return:
    """

    # get data from file names
    files = glob.glob(os.path.join(dir_path, fname_exp))
    prefixes = []
    channels = []
    times = []
    positions = []
    slices = []
    for f in files:
        name_root, _ = os.path.splitext(os.path.basename(f))
        m = re.match(exp, name_root)

        try:
            prefixes.append(m.group("prefix"))
        except:
            prefixes.append("")

        try:
            channels.append(int(m.group("channel")))
        except:
            channels.append(0)

        try:
            times.append(int(m.group("time")))
        except:
            times.append(0)

        try:
            positions.append(int(m.group("position")))
        except:
            positions.append(0)

        try:
            slices.append(int(m.group("slice")))
        except:
            slices.append(0)

    # parameter sizes
    nz = np.max(slices) + 1
    nc = np.max(channels) + 1
    nt = np.max(times) + 1
    nxy = np.max(positions) + 1

    if nxy > 1:
        raise Exception("Not implemented for nxy != 1")

    # read image to get image size
    im_first, _, _ = read_tiff(files[0], dtype=dtype)
    _, ny, nx = im_first.shape

    # combine images to one array
    imgs = np.zeros((nc, nt, nz, ny, nx)) * np.nan
    for ii in range(len(files)):
        img, _, _, = read_tiff(files[ii], dtype=dtype)
        imgs[channels[ii], times[ii], slices[ii]] = img[0]

    if np.any(np.isnan(imgs)):
        print("WARNING: not all channels/times/slices/positions were accounted for. Those that were not found are replaced by NaNs")

    # save results
    save_tiff(imgs, fname_out, dtype='float32', axes_order="CTZYX", hyperstack=True)

# file naming
def get_unique_name(fname, mode='file'):
    """
    Produce a unique filename by appending an integer

    :param fname:
    :param mode: 'file' or 'dir'
    :return:
    """
    if not os.path.exists(fname):
        return fname

    if mode == 'file':
        fname_root, ext = os.path.splitext(fname)
    elif mode == 'dir':
        fname_root = fname
        ext = ''
    else:
        raise Exception("'mode' must be 'file' or 'dir', but was '%s'" % mode)

    ii = 1
    while os.path.exists(fname):
        fname = '%s_%d%s' % (fname_root, ii, ext)
        ii += 1

    return fname

def get_timestamp():
    now = datetime.datetime.now()
    return '%04d_%02d_%02d_%02d;%02d;%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

# image processing
def azimuthal_avg(img, dist_grid, bin_edges, weights=None):
    """
    Take azimuthal average of img. All points which have a dist_grid value lying
    between successive bin_edges will be averaged. Points are considered to lie within a bin
    if their value is strictly smaller than the upper edge, and greater than or equal to the lower edge.
    :param np.array img: 2D image
    :param np.array dist_grid:
    :param np.array or list bin_edges:
    :param np.array weights:

    :return az_avg:
    :return sdm:
    :return dist_mean:
    :return dist_sd:
    :return npts_bin:
    :return masks:
    """

    # there are many possible approaches for doing azimuthal averaging. Naive way: for each mask az_avg = np.mean(img[mask])
    # also can do using scipy.ndimage.mean(img, labels=masks, index=np.arange(0, n_bins). scipy approach is slightly slower
    # than np.bincount. Naive approach ~ factor of 2 slower.


    if weights is None:
        weights = np.ones(img.shape)

    n_bins = len(bin_edges) - 1
    # build masks. initialize with integer value that does not conflict with any of our bins
    masks = np.ones((img.shape[0], img.shape[1]), dtype=np.int) * n_bins
    for ii in range(n_bins):
        # create mask
        bmin = bin_edges[ii]
        bmax = bin_edges[ii + 1]
        mask = np.logical_and(dist_grid < bmax, dist_grid >= bmin)
        masks[mask] = ii

    # get indices to use during averaging. Exclude any nans in img, and exclude points outside of any bin
    to_use_inds = np.logical_and(np.logical_not(np.isnan(img)), masks < n_bins)
    npts_bin = np.bincount(masks[to_use_inds])

    # failing to correct for case where some points are not contained in any bins. These had the same bin index as
    # the first bin, which caused problems!
    # nan_inds = np.isnan(img)
    # npts_bin = np.bincount(masks.ravel(), np.logical_not(nan_inds).ravel())
    # set any points with nans to zero, and these will be ignored by averaging due to above correction of npts_bin
    # img[nan_inds] = 0
    # dist_grid[nan_inds] = 0
    # az_avg = np.bincount(masks.ravel(), img.ravel())[0:-1] / npts_bin
    # sd = np.sqrt(np.bincount(masks.ravel(), img.ravel() ** 2) / npts_bin - az_avg ** 2) * np.sqrt(npts_bin / (npts_bin - 1))
    # dist_mean = np.bincount(masks.ravel(), dist_grid.ravel()) / npts_bin
    # dist_sd = np.sqrt(np.bincount(masks.ravel(), dist_grid.ravel() ** 2) / npts_bin - dist_mean ** 2) * np.sqrt(npts_bin / (npts_bin - 1))

    # do azimuthal averaging
    az_avg = np.bincount(masks[to_use_inds], img[to_use_inds]) / npts_bin
    # correct variance for unbiased estimator. (of course still biased for sd)
    sd = np.sqrt(np.bincount(masks[to_use_inds], img[to_use_inds] ** 2) / npts_bin - az_avg ** 2) * np.sqrt(npts_bin / (npts_bin - 1))
    dist_mean = np.bincount(masks[to_use_inds], dist_grid[to_use_inds]) / npts_bin
    dist_sd = np.sqrt(np.bincount(masks[to_use_inds], dist_grid[to_use_inds] ** 2) / npts_bin - dist_mean ** 2) * np.sqrt(npts_bin / (npts_bin - 1))

    # pad to match expected size given number of bin edges provided
    n_occupied_bins = npts_bin.size
    extra_zeros = np.zeros(n_bins - n_occupied_bins)
    if n_occupied_bins < n_bins:
        npts_bin = np.concatenate((npts_bin, extra_zeros), axis=0)
        az_avg = np.concatenate((az_avg, extra_zeros * np.nan), axis=0)
        sd = np.concatenate((sd, extra_zeros * np.nan), axis=0)
        dist_mean = np.concatenate((dist_mean, extra_zeros * np.nan), axis=0)
        dist_sd = np.concatenate((dist_sd, extra_zeros * np.nan), axis=0)

    # alternate approach with scipy.ndimage functions. 10-20% slower in my tests
    # az_avg = ndimage.mean(img, labels=masks,  index=np.arange(0, n_bins))
    # sd = ndimage.standard_deviation(img, labels=masks, index=np.arange(0, n_bins))
    # dist_mean = ndimage.mean(dist_grid, labels=masks, index=np.arange(0, n_bins))
    # dist_sd = ndimage.standard_deviation(dist_grid, labels=masks, index=np.arange(0, n_bins))
    # npts_bin = ndimage.sum(np.ones(img.shape), labels=masks, index=np.arange(0, n_bins))

    sdm = sd / np.sqrt(npts_bin)

    return az_avg, sdm, dist_mean, dist_sd, npts_bin, masks

def elliptical_grid(params, xx, yy, units='mean'):
    """
    Get elliptical `distance' grid for use with azimuthal averaging. These `distances' will be the same for points lying
    on ellipses with the parameters specified by params.

    Ellipse equation is (x - cx) ^ 2 / A ^ 2 + (y - cy) ^ 2 / B ^ 2 = 1
    Define d_A  = sqrt((x - cx) ^ 2 + (y - cy) ^ 2 * (A / B) ^ 2)...which is the
    Define d_B  = sqrt((x - cx) ^ 2 * (B / A) ^ 2 + (y - cy) ^ 2) = (B / A) * d_A
    Define d_AB = sqrt((x - cx) ^ 2 * (B / A) + (y - cy) ^ 2 * (A / B)) = sqrt(B / A) * d_A                                                      %
    for a given ellipse, d_A is the distance along the A axis, d_B along the B
    axis, and d_AB along 45 deg axis.i.e.d_A(x, y) gives the length of the A
    axis of an ellipse with the given axes A and B that contains (x, y).

    :param params: [cx, cy, aspect_ratio, theta]. aspect_ratio = wy/wx. theta is the rotation angle of the x-axis of the
    ellipse measured CCW from the x-axis of the coordinate system
    :param xx: x-coordinates to compute grid on
    :param yy: y-coordinates to compute grid on
    :param units: 'mean', 'major', or 'minor'
    :return:
    """

    cx = params[0]
    cy = params[1]
    aspect_ratio = params[2]
    theta = params[3]

    distance_grid = np.sqrt(
        ((xx - cx) * np.cos(theta) - (yy - cy) * np.sin(theta))**2 +
        ((yy - cy) * np.cos(theta) + (xx - cx) * np.sin(theta))**2 * aspect_ratio**2)


    if aspect_ratio < 1:
        if units == 'minor':
            pass # if aspect ratio < 1 we are already in 'minor' units.
        elif units == 'major':
            distance_grid = distance_grid / aspect_ratio
        elif units == 'mean':
            distance_grid = distance_grid / np.sqrt(aspect_ratio)
        else:
            raise Exception("'units' must be 'minor', 'major', or 'mean', but was '%s'" % units)
    else:
        if units == 'minor':
            distance_grid = distance_grid / aspect_ratio
        elif units == 'major':
            pass # if aspect ratio > 1 we are already in 'major' units
        elif units == 'mean':
            distance_grid = distance_grid / np.sqrt(aspect_ratio)
        else:
            raise Exception("'units' must be 'minor', 'major', or 'mean', but was '%s'" % units)

    return distance_grid

def bin(img, bin_size, mode='sum'):
    """
    bin image by combining adjacent pixels

    In 1D, this is a straightforward problem. The image is a vector,
    I = (I[0], I[1], ..., I[nx-1])
    and the binning operator is a nx/nx_bin x nx matrix
    M = [[1, 1, ..., 1, 0, ..., 0, 0, ..., 0]
         [0, 0, ..., 0, 1, ..., 1, 0, ..., 0]
         ...
         [0, ...,              0,  1, ..., 1]]
    which has a tensor product structure, which is intuitive because we are operating on each run of x points independently.
    M = identity(nx/nx_bin) \prod ones(1, nx_bin)
    the binned image is obtained from matrix multiplication
    Ib = M * I

    In 2D, this situation is very similar. Here we take the image to be a row stacked vector
    I = (I[0, 0], I[0, 1], ..., I[0, nx-1], I[1, 0], ..., I[ny-1, nx-1])
    the binning operator is a (nx/nx_bin)*(ny/ny_bin) x nx*ny matrix which has a tensor product structure.

    This time the binning matrix has dimension (nx/nx_bin * ny/ny_bin) x (nx * ny)
    The top row starts with nx_bin 1's, then zero until position nx, and then ones until position nx + nx_bin.
    This pattern continues, with nx_bin 1's starting at jj*nx for jj = 0,...,ny_bin-1. The second row follows a similar
    pattern, but shifted by nx_bin pixels
    M = [[1, ..., 1, 0, ..., 0, 1, ..., 1, 0,...]
         [0, ..., 0, 1, ..., 1, ...
    Again, this has tensor product structure. Notice that the first (nx/nx_bin) x nx entries are the same as the 1D case
    and the whole matrix is constructed from blocks of these.
    M = [identity(ny/ny_bin) \prod ones(1, ny_bin)] \prod  [identity(nx/nx_bin) \prod ones(1, nx_bin)]

    Again, Ib = M*I

    Probably this pattern generalizes to higher dimensions!

    :param img: image to be binned
    :param nbin: [ny_bin, nx_bin] where these must evenly divide the size of the image
    :param mode: either 'sum' or 'mean'
    :return:
    """
    # todo: could also add ability to bin in this direction. Maybe could simplify function by allowing binning in
    # arbitrary dimension (one mode), with another mode to bin only certain dimensions and leave others untouched.
    # actually probably don't need to distinguish modes, this can be done by looking at bin_size.
    # still may need different implementation for the cases, as no reason to flatten entire array to vector if not
    # binning. But maybe this is not really a performance hit anyways with the sparse matrices?

    # if three dimensional, bin each image
    if img.ndim == 3:
        ny_bin, nx_bin = bin_size
        nz, ny, nx = img.shape

        # size of image after binning
        nx_s = int(nx / nx_bin)
        ny_s = int(ny / ny_bin)

        m_binned = np.zeros((nz, ny_s, nx_s))
        for ii in range(nz):
            m_binned[ii, :] = bin(img[ii], bin_size, mode=mode)

    # bin 2D image
    elif img.ndim == 2:
        ny_bin, nx_bin = bin_size
        ny, nx = img.shape

        if ny % ny_bin != 0 or nx % nx_bin != 0:
            raise Exception('bin size must evenly divide image size.')

        # size of image after binning
        nx_s = int(nx/nx_bin)
        ny_s = int(ny/ny_bin)

        # matrix which performs binning operation on row stacked matrix
        # need to use sparse matrices to bin even moderately sized images
        bin_mat_x = sp.kron(sp.identity(nx_s), np.ones((1, nx_bin)))
        bin_mat_y = sp.kron(sp.identity(ny_s), np.ones((1, ny_bin)))
        bin_mat_xy = sp.kron(bin_mat_y, bin_mat_x)

        # row stack img. img.ravel() = [img[0, 0], img[0, 1], ..., img[0, nx-1], img[1, 0], ...]
        m_binned = bin_mat_xy.dot(img.ravel()).reshape([ny_s, nx_s])

        if mode == 'sum':
            pass
        elif mode == 'mean':
            m_binned = m_binned / (nx_bin * ny_bin)
        else:
            raise Exception("mode must be either 'sum' or 'mean' but was '%s'" % mode)

    # 1D "image"
    elif img.ndim == 1:

        nx_bin = bin_size[0]
        nx = img.size

        if nx % nx_bin != 0:
            raise Exception('bin size must evenly divide image size.')
        nx_s = int(nx / nx_bin)

        bin_mat_x = sp.kron(sp.identity(nx_s), np.ones((1, nx_bin)))
        m_binned = bin_mat_x.dot(img)

        if mode == 'sum':
            pass
        elif mode == 'mean':
            m_binned = m_binned / nx_bin
        else:
            raise Exception("mode must be either 'sum' or 'mean' but was '%s'" % mode)

    else:
        raise Exception("Only 1D, 2D, or 3D arrays allowed")

    return m_binned

def toeplitz_filter_mat(filter, img_size, mode='valid'):
    """
    Return the Toeplitz matrix which performs a given 1D or 2D filter on a 1D signal or 2D image. We assume the
    image is represented as a row-stacked vector (i.e. [m[0, 0], m[0, 1], ..., m[0, nx-1], m[1, 0], ..., m[ny-1, nx-1]])

    For large images, generally a better idea to use scipy.convolve or scipy.convolve2d instead

    Note that this technique cannot perform binning, although it is closely related. It can perform a rolling average.

    todo: could add 'same' mode, which keeps input same size. And enforce only for odd sized filters.
    :param filter:
    :param img_size: [nx, ny] or [nx]
    :param mode: 'valid' or 'full'
    :return:

    ################
    1D case
    ################
    'valid' = regions of full overlap
    mat = [ f[0], f[1], f[2], ..., f[n],   0, ...,  0  ]
          [  0  , f[0], f[1], ...      , f[n], 0,  ... ]
          [                   ...                      ]
          [ ...                                   f[n] ]

    'full':
    mat = [  f[n] ,   0 , ...                     ]
          [ f[n-1], f[n], 0, ...                  ]
          [ ...                                   ]
          [ ...                              f[0] ]

    ################
    2D case
    ################
    here fy[i] is a matrix representing a 1D type filter for y = i
    mat = [ fy[0], fy[1], fy[2], ..., fy[n],   0  , ...,  0   ]
          [  0  , fy[0], fy[1], ...        , fy[n],  0 ,  ... ]
          [                     ...                           ]
          [   ...                                       fy[n] ]
    """

    if filter.ndim == 1:
        nx = img_size[0]

        if mode == 'valid':
            first_row = np.pad(filter, (0, nx - filter.size), mode='constant', constant_values=0)
            first_col = np.zeros(nx - filter.size + 1)
            first_col[0] = first_row[0]
        elif mode == 'full':
            first_row = np.zeros(nx)
            first_row[0] = filter[-1]

            first_col = np.pad(np.flip(filter), (0, nx - 1), mode='constant', constant_values=0)
        else:
            raise Exception("mode must be 'valid' or 'full' but was '%s'" % mode)

        # todo: might want to try and generate directly as sparse matrix
        mat = sp.csc_matrix(scipy.linalg.toeplitz(first_col, r=first_row))

    elif filter.ndim == 2:
        ny, nx = img_size
        ny_filter, nx_filter = filter.shape

        # get 1D filter blocks
        filters_1d = []
        for row in filter:
            filters_1d.append(toeplitz_filter_mat(row, [nx], mode=mode))

        # combine blocks to full matrix
        ny_block, _ = filters_1d[0].shape
        if mode == 'valid':
            # in this case, the x-filter blocks have size (nx - filter.shape[1]) x nx
            # and these appear on the diagonals of a matrix of the size (ny - filter.shape[0]) x ny
            skeleton_mat_ny = ny - ny_filter + 1

            mat = sp.csc_matrix((ny_block * skeleton_mat_ny, nx * ny))
            for jj in range(filter.shape[0]):
                diag_ones = sp.diags([1], jj, shape=(skeleton_mat_ny, ny))
                mat += sp.kron(diag_ones, filters_1d[jj])

        elif mode == 'full':

            skeleton_mat_ny = ny + ny_filter - 1
            mat = sp.csc_matrix((ny_block * skeleton_mat_ny, nx * ny))
            for jj in range(filter.shape[0]):
                diag_ones = sp.diags([1], -jj, shape=(skeleton_mat_ny, ny))
                mat += sp.kron(diag_ones, filters_1d[jj])

    return mat

# background estimation
def estimate_background(img):
    """
    Estimate camera offset background of uint16 image from histogram.

    :param img:

    :return bg: Background value.
    :return fit_params: fit parameters of half-gaussian fit to histogram. See fit_half_gauss1d().
    """

    # estimate maximum coarsely
    bin_edges = np.linspace(0, 2**16, 1000)
    h, bin_edges = np.histogram(img, bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bg_coarse = bin_edges[np.argmax(h)]

    # finer estimate
    bin_edges = np.arange(0, 3*bg_coarse, 2)
    h, bin_edges = np.histogram(img, bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # fit to gaussian
    fit_results, fit_fn = fit_half_gauss1d(h, x=bin_centers, init_params=[None, None, None, 0, None, None],
                                           fixed_params=[0, 0, 0, 1, 0, 0],
                                           bounds=((0, bin_centers.min(), 0, 0, 0, 0),
                                                   (np.inf, bin_centers.max(), bin_centers.max() - bin_centers.min(),
                                                    np.inf, bin_centers.max() - bin_centers.min(), np.inf)))

    fparams = fit_results['fit_params']
    bg = fparams[1]

    return bg, fparams

# resampling functions
# todo: swap resample/expand names as they are not accurate!
def resample(img, nx=2, ny=2):
    """
    Resample image by expanding each pixel into a rectangle of ny x nx identical pixels

    :param img: image to resample
    :param nx:
    :param ny:
    :return:
    """
    if not isinstance(nx, int) or not isinstance(ny, int):
        raise Exception('nx and ny must be ints')

    block = np.ones((ny, nx))
    img_resampled = np.kron(img, block)

    return img_resampled

def resample_fourier_sp(img_ft, mx=2, my=2, centered=True):
    """
    Resample the Fourier transform of image. In real space, this operation corresponds to replacing each pixel with a
    myxmx square of identical pixels. Note that this is often NOT the desired resampling behavior if you e.g. have
    an image. In that case you should use expand_fourier_sp() instead.


    In Fourier space we take advantage of the following connection. Let a be the original sequence and
    b[2n] = b[2n+1] = a[n]
    a[k] = \sum_{n=0}^{N-1} a[n] * exp(-2*pi*i/N * k * n)
    b[k] = \sum_{n=0}^{N-1} a[n] * {exp[-2*pi*i/(2N) * k * 2n] + exp[-2*pi*i/(2N) * k * 2n]}
         =  (1 + exp(-2*pi*i*k/(2N)) a[k]
    For N-1 < k < 2N, we are free to replace k by k-N

    This generalizes to
    b[k] = a[k] * \sum_{l=0}^{m-1} exp(-2*pi*i*k/(mN) * l)
         = a[k] * (1 - exp(-2*pi*i*k/N)) / (1 - exp(-2*pi*i*k/(mN)))

    :param img_ft:
    :param centered: If False, treated as raw output of fft.fft2. If true, treated as fft.fftshift(fft.fft2())
    :return:
    """

    ny, nx = img_ft.shape

    if centered:
        img_ft = fft.ifftshift(img_ft)

    kxkx, kyky = np.meshgrid(range(nx*mx), range(ny*my))

    phase_x = np.exp(-1j*2*np.pi * kxkx / (nx*mx))
    factor_x = (1 - phase_x**mx) / (1 - phase_x)
    # at kx or ky = 0 these give indeterminate forms
    factor_x[kxkx == 0] = mx

    phase_y = np.exp(-1j*2*np.pi * kyky / (ny*my))
    factor_y = (1 - phase_y**my) / (1 - phase_y)
    factor_y[kyky == 0] = my

    img_ft_resampled = factor_x * factor_y * np.tile(img_ft, (my, mx))

    if centered:
        img_ft_resampled = fft.fftshift(img_ft_resampled)

    return img_ft_resampled

def expand_im(img, mx=2, my=2):
    """
    Expand real-space imaging while keeping fourier content constant

    :param img: original image
    :param mx: factor to expandd along the x-direction
    :param my: factor to expand along the y-direction

    :return img_resampled:
    """
    img_resampled = fft.ifft2(expand_fourier_sp(fft.fft2(img), mx=mx, my=my, centered=False))
    return img_resampled

def expand_fourier_sp(img_ft, mx=2, my=2, centered=True):
    """
    Expand image by factors of mx and my while keeping Fourier content constant.

    :param img_ft: frequency space representation of image
    :param mx: factor to resample image along the x-direction
    :param my: factor to resample image along the y-direction
    :param centered: if True assume img_ft has undergone an fftshift so that zero-frequency components are in the center

    :return img_ft_expanded:
    """

    if mx == 1 and my == 1:
        return img_ft

    ny, nx = img_ft.shape
    fx_old = get_fft_frqs(nx)
    fy_old = get_fft_frqs(ny)

    fx_new = get_fft_frqs(nx * mx, dt=0.5)
    fy_new = get_fft_frqs(ny * my, dt=0.5)

    if not centered:
        img_ft = fft.fftshift(img_ft)

    ix_start = np.where(fx_new == fx_old[0])[0][0]
    iy_start = np.where(fy_new == fy_old[0])[0][0]

    img_ft_exp = np.zeros((ny * my, nx * mx), dtype=np.complex)
    img_ft_exp[iy_start : iy_start + ny, ix_start:ix_start + nx] = img_ft

    # todo: if even array has one extra negative frequency. Now I've added its positive freq
    # partner, so shouldn't I have to put complex conjugate of that value in partner spot?
    # yes, and divide by 2!
    if np.mod(nx, 2) == 0:
        ix_ops = np.where(fx_new == -fx_old[0])[0][0]
        img_ft_exp[:, ix_start] = 0.5 * img_ft_exp[:, ix_start]
        img_ft_exp[:, ix_ops] = img_ft_exp[:, ix_start]

    if np.mod(ny, 2) == 0:
        iy_ops = np.where(fy_new == - fy_old[0])[0][0]
        img_ft_exp[iy_start, :] = 0.5 * img_ft_exp[iy_start, :]
        img_ft_exp[iy_ops, :] = img_ft_exp[iy_start, :]

    # but will have double corrected for [iy_start, ix_start], [iy_start, ix_ops], [iy_ops, ix_start], and [iy_ops, ix_ops]
    if np.mod(ny, 2) == 0 and np.mod(nx, 2) == 0:
        img_ft_exp[iy_start, ix_start] = 2 * img_ft_exp[iy_start, ix_start]
        img_ft_exp[iy_start, ix_ops] = 2 * img_ft_exp[iy_start, ix_ops]
        img_ft_exp[iy_ops, ix_start] = 2 * img_ft_exp[iy_ops, ix_start]
        img_ft_exp[iy_ops, ix_ops] = 2 * img_ft_exp[iy_ops, ix_ops]

    # correct normalization
    img_ft_exp = mx * my * img_ft_exp

    if not centered:
        img_ft_exp = fft.ifftshift(img_ft_exp)

    return img_ft_exp

# geometry tools
def get_peak_value(img, x, y, peak_coord, peak_pixel_size=1):
    """
    Estimate value for a peak that is not precisely aligned to the pixel grid by performing a weighted average
    over neighboring pixels, based on how much these overlap with a rectangular area surrounding the peak.
    The size of this rectangular area is set by peak_pixel_size, given in integer multiples of a pixel.

    :param img: image containing peak
    :param x: x-coordinates of image
    :param y: y-coordinates of image
    :param peak_coord: peak coordinate [px, py]
    :param peak_pixel_size: number of pixels (along each direction) to sum to get peak value
    :return peak_value: estimated value of the peak
    """
    px, py = peak_coord

    # frequency coordinates
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xx, yy = np.meshgrid(x, y)

    # find closest pixel
    ix = np.argmin(np.abs(px - x))
    iy = np.argmin(np.abs(py - y))

    # get ROI around pixel for weighted averaging
    roi = get_centered_roi([iy, ix], [3 * peak_pixel_size, 3 * peak_pixel_size])
    img_roi = img[roi[0]:roi[1], roi[2]:roi[3]]
    xx_roi = xx[roi[0]:roi[1], roi[2]:roi[3]]
    yy_roi = yy[roi[0]:roi[1], roi[2]:roi[3]]

    # estimate value from weighted average of pixels in ROI, based on overlap with pixel area centered at [px, py]
    weights = np.zeros(xx_roi.shape)
    for ii in range(xx_roi.shape[0]):
        for jj in range(xx_roi.shape[1]):
            weights[ii, jj] = pixel_overlap([py, px], [yy_roi[ii, jj], xx_roi[ii, jj]],
                                                  [peak_pixel_size * dy, peak_pixel_size * dx], [dy, dx]) / (dx * dy)

    peak_value = np.average(img_roi, weights=weights)

    return peak_value

def pixel_overlap(centers1, centers2, lens1, lens2=None):
    """
    Calculate overlap of two nd-square pixels. The pixels go from coordinates
    centers[ii] - 0.5 * lens[ii] to centers[ii] + 0.5 * lens[ii].

    :param centers1: list of coordinates defining centers of first pixel along each dimension
    :param centers2: list of coordinates defining centers of second pixel along each dimension
    :param lens1: list of pixel 1 sizes along each dimension
    :param lens2: list of pixel 2 sizes along each dimension
    :return overlaps: overlap area of pixels
    """

    if not isinstance(centers1, list):
        centers1 = [centers1]

    if not isinstance(centers2, list):
        centers2 = [centers2]

    if not isinstance(lens1, list):
        lens1 = [lens1]

    if lens2 is None:
        lens2 = lens1

    if not isinstance(lens2, list):
        lens2 = [lens2]

    overlaps = []
    for c1, c2, l1, l2 in zip(centers1, centers2, lens1, lens2):
        if np.abs(c1 - c2) >= 0.5*(l1 + l2):
            overlaps.append(0)
        else:
            # ensure whichever pixel has leftmost edge is c1
            if (c1 - 0.5 * l1) > (c2 - 0.5 * l2):
                c1, c2 = c2, c1
                l1, l2 = l2, l1
            # by construction left start of overlap is c2 - 0.5*l2
            # end is either c2 + 0.5 * l2 OR c1 + 0.5 * l1
            lstart = c2 - 0.5 * l2
            lend = np.min([c2 + 0.5 * l2, c1 + 0.5 * l1])
            overlaps.append(np.max([lend - lstart, 0]))

    return np.prod(overlaps)

def segment_intersect(start1, end1, start2, end2):
    """
    Get intersection point of two 2D line segments
    :param start1: [x, y]
    :param end1:
    :param start2:
    :param end2:
    :return:
    """
    # solve S1 * (1-t) + e1 * t = S2 * (1-r) * e2 * r
    # phrase this as roi_size matrix problem:
    # S1 - S2 = [[e1_x - s1_x, e2_x - s2_x]; [e1_y - s1_y, e2_y - s2_y]] * [t; r]
    start1 = np.asarray(start1)
    end1 = np.asarray(end1)
    start2 = np.asarray(start2)
    end2 = np.asarray(end2)

    try:
        # solve system of equations by inverting matrix
        M = np.array([[start1[0] - end1[0], end2[0] - start2[0]],
                      [start1[1] - end1[1], end2[1] - start2[1]]])
        vs = np.linalg.inv(M).dot(np.asarray([[start1[0] - start2[0]], [start1[1] - start2[1]]]))
    except np.linalg.LinAlgError:
        return None

    t = vs[0][0]
    r = vs[1][0]

    # check within bounds
    if t<=1 and t>=0 and r<=1 and r>=0:
        return start1 * (1-t) + end1 * t
    else:
        return None

# working with regions of interest
def get_extent(y, x):
    """
    Get extent required for plotting arrays using imshow in real coordinates. The resulting list can be
    passed directly to imshow using the extent keyword.

    Here we assume the values y and x are equally spaced and describe the center coordinate of each pixel

    :param y: equally spaced y-coordinates
    :param x: equally spaced x-coordinates
    :return extent: [xstart, xend, yend, ystart]
    """
    dy = y[1] - y[0]
    dx = x[1] - x[0]
    extent = [x[0] - 0.5 * dx, x[-1] + 0.5 * dx, y[-1] + 0.5 * dy, y[0] - 0.5 * dy]
    return extent

def roi2full(coords_roi, roi):
    """
    coords_roi = [c1, c2, ...]
    roi = [c1start, c1end, c2start, c2end, ...]
    :param coords_roi:
    :param roi:
    :return:
    """
    coords_full = []
    for ii, c in enumerate(coords_roi):
        coords_full.append(roi[2*ii] + c)

    return coords_full

def full2roi(coords_full, roi):
    """
    coords_full = [c1, c2, ...]
    roi = [c1start, c1end, c2start, c2end, ...]

    :param coords_full:
    :param roi:
    :return:
    """

    coords_roi = []
    for ii, c in enumerate(coords_full):
        coords_roi.append(c - roi[2*ii])

    return coords_roi

def get_centered_roi(centers, sizes):
    """
    Get end points of an roi centered about centers (as close as possible) with length sizes.
    If the ROI size is odd, the ROI will be perfectly centered. Otherwise, the centering will
    be approximation

    roi = [start_0, end_0, start_1, end_1, ..., start_n, end_n]

    Slicing an array as A[start_0:end_0, start_1:end_1, ...] gives the desired ROI.
    Note that following python array indexing convention end_i are NOT contained in the ROI

    :param centers: list of centers [c1, c2, ..., cn]
    :param sizes: list of sizes [s1, s2, ..., sn]
    :return roi: [start_0, end_0, start_1, end_1, ..., start_n, end_n]
    """
    roi = []
    for c, n in zip(centers, sizes):

        # get ROI closest to centered
        end_test = np.round(c + (n - 1) / 2) + 1
        end_err = np.mod(end_test, 1)
        start_test = np.round(c - (n - 1) / 2)
        start_err = np.mod(start_test, 1)

        if end_err > start_err:
            start = start_test
            end = start + n
        else:
            end = end_test
            start = end - n

        roi.append(int(start))
        roi.append(int(end))

    return roi

def crop_roi(roi, image_size):
    """
    Crop ROI so it fits in an image of roi_size given size.
    :param roi:
    :param image_size:
    :return:
    """
    cropped_roi = []
    for ii, n in enumerate(image_size):

        start = roi[2*ii]
        if start < 0:
            start = 0

        end = roi[2*ii+1]
        if end > n:
            end = n

        cropped_roi.append(start)
        cropped_roi.append(end)

    return cropped_roi

def map_intervals(vals, from_intervals, to_intervals):
    """
    Given value v in interval [a, b], find the corresponding value in the interval [c, d]

    :param vals: list of vals [v1, v2, v3, ..., vn]
    :param from_intervals: list of intervals containing start values [[a1, b1], [a2, b2], ..., [an, bn]]
    :param to_intervals: list of intervals containing end valus [[c1, d1], [c2, d2], ..., [cn, dn]]
    :return:
    """
    if not isinstance(vals, list):
        vals = [vals]

    if not isinstance(from_intervals[0], list):
        from_intervals = [from_intervals]

    if not isinstance(to_intervals[0], list):
        to_intervals = [to_intervals]

    vals_out = []
    for v, i1, i2 in zip(vals, from_intervals, to_intervals):
        vals_out.append( (v - i1[0]) * (i2[1] - i2[0]) / (i1[1] - i1[0])  + i2[0])

    return vals_out

# fitting tools
def get_moments(img, order=1, coords=None, dims=None):
    """
    Calculate moments of distribution of arbitrary size
    :param img:
    :param order: order of moments to be calculated
    :param coords: list of coordinate arrays for each dimension e.g. [y, x], where y, x etc. are 1D arrays
    :param dims: dimensions to be summed over. For example, given roi_size 3D array of size Nz x Ny x Nz, calculate the 2D
    moments of each slice by setting dims = [1, 2]
    :return:
    """
    if dims is None:
        dims = range(img.ndim)

    if coords is None:
        coords = [np.arange(s) for ii, s in enumerate(img.shape) if ii in dims]
    # ensure coords are float arrays to avoid overflow issues
    coords = [np.array(c, dtype=np.float) for c in coords]

    if len(dims) != len(coords):
        raise Exception('dims and coordinates must have the same length')

    # weight summing only over certain dimensions
    w = np.nansum(img, axis=tuple(dims), dtype=np.float)

    # as trick to avoid having to meshgrid any of the coordinates, we can use NumPy's array broadcasting. Because this
    # looks at the trailing array dimensions, we need to swap our desired axis to be the last dimension, multiply by the
    # coordinates to do the broadcasting, and then swap back
    moments = [np.nansum(np.swapaxes(np.swapaxes(img, ii, img.ndim-1) * c**order, ii, img.ndim-1),
               axis=tuple(dims), dtype=np.float) / w
               for ii, c in zip(dims, coords)]

    return moments

def poly(xx, yy, params, max_orders=(1, 1)):
    """

    :param xx:
    :param yy:
    :param params: [cx, cy, b00, b01, ..., b0N, b10, ...bMN], where
    P(x) = \sum_{n,m} cnm (X - cx)**n * (Y - cy)**m
    :param max_orders:
    :return:
    """

    max_x, max_y = max_orders
    ny = max_y + 1
    nx = max_x + 1
    norders = nx * ny

    if len(params) != (norders + 2):
        raise Exception('params is not equal to norders')

    cx = params[0]
    cy = params[1]

    val = 0
    for ii in range(nx):
        for jj in range(ny):
            val += params[2 + ii * ny + jj] * (xx - cx) ** ii * (yy - cy) ** jj

    return val

def poisson(n, p):
    """
    Poisson distribution

    :param n: number parameter. This function will also return sensible values for non-integer values
    :param p: [A, center, mean, bg]
    :return value:
    """
    return p[0] * scipy.stats.poisson.pmf(n, p[2]) + p[3]

def gauss_fn(x, y, p):
    """
    Rotated 2D gaussian function. The angle theta is defined clockwise from the x- (or y-) axis. NOTE: be careful
    with this when looking at results using e.g. matplotlib.imshow, as this will display the negative y-axis on top.

    :param x: x-coordinates to evaluate function at.
    :param y: y-coordinates to evaluate function at. Either same size as x, or broadcastable with x.
    :param p: [A, cx, cy, sxrot, syrot, bg, theta]
    :return value:
    """
    if len(p) != 7:
        raise Exception("parameter list p must have length 7")

    xrot = np.cos(p[6]) * (x - p[1]) - np.sin(p[6]) * (y - p[2])
    yrot = np.cos(p[6]) * (y - p[2]) + np.sin(p[6]) * (x - p[1])
    return p[0] * np.exp(-xrot ** 2 / (2 * p[3] ** 2) - yrot ** 2 / (2 * p[4] ** 2)) + p[5]

def gauss_jacobian(x, y, p):
    """
    Jacobian of gauss_fn

    :param x:
    :param y:
    :param p: [A, cx, cy, sx, sy, bg, theta]
    :return value:
    """
    if len(p) != 7:
        raise Exception("parameter list p must have length 7")

    # useful functions that show up in derivatives
    xrot = np.cos(p[6]) * (x - p[1]) - np.sin(p[6]) * (y - p[2])
    yrot = np.cos(p[6]) * (y - p[2]) + np.sin(p[6]) * (x - p[1])
    exps = np.exp(-xrot**2 / (2 * p[3] ** 2) -yrot**2 / (2 * p[4] ** 2))

    bcast_shape = (x + y).shape

    return [exps,
            p[0] * exps * (xrot / p[3]**2 * np.cos(p[6]) + yrot / p[4]**2 * np.sin(p[6])),
            p[0] * exps * (yrot / p[4]**2 * np.cos(p[6]) - xrot / p[3]**2 * np.sin(p[6])),
            p[0] * exps * xrot ** 2 / p[3] ** 3,
            p[0] * exps * yrot ** 2 / p[4] ** 3,
            np.ones(bcast_shape),
            p[0] * exps * xrot * yrot * (1 / p[3]**2 - 1 / p[4]**2)]

def ngauss_fn(x, y, p):
    """
    Sum of n 2D gaussians
    :param x:
    :param y:
    :param p: [A1, cx1, cx2, sx1, sx2, theta1, A2, ..., thetan, bg]
    :return:
    """
    if len(p) % 6 != 1:
        raise Exception("Parameters")

    ngaussians = (len(p) - 1) // 6

    val = 0
    for ii in range(ngaussians - 1):
        ps = np.concatenate((np.array(p[6*ii: 6*ii + 5]), np.array([0]), np.atleast_1d([p[ii * 6 + 5]])))
        val += gauss_fn(x, y, ps)

    # deal with last gaussian, which also gets background term
    ps = np.concatenate((np.array(p[-7:-2]), np.atleast_1d(p[-1]), np.atleast_1d(p[-2])))
    val += gauss_fn(x, y, ps)
    return val

def ngauss_jacobian(x, y, p):
    """
    Jacobian of the sum of n 2D gaussians
    :param x:
    :param y:
    :param p:
    :return:
    """
    if len(p) % 6 != 1:
        raise Exception("Parameters")

    ngaussians = (len(p) - 1) // 6

    jac_list = []
    for ii in range(ngaussians - 1):
        ps = np.concatenate((np.array(p[6 * ii: 6 * ii + 5]), np.array([0]), np.atleast_1d([p[ii * 6 + 5]])))
        jac_current = gauss_jacobian(x, y, ps)
        jac_list += jac_current[:-2] + [jac_current[-1]]

    # deal with last gaussian, which also gets background term
    ps = np.concatenate((np.array(p[-7:-2]), np.atleast_1d(p[-1]), np.atleast_1d(p[-2])))
    jac_current = gauss_jacobian(x, y, ps)
    jac_list += jac_current[:-2] + [jac_current[-1]] + [jac_current[-2]]

    return jac_list

def fit_half_gauss1d(y, init_params=None, fixed_params=None, sd=None, x=None, bounds=None):
    """
    Fit function that has two Gaussian halves with different sigmas and offsets but match smoothly at cx

    :param y:
    :param init_params: [A1, cx, sx1, bg1, sx2, bg2]
    :param fixed_params:
    :param sd:
    :param x:
    :param bounds:
    :return:
    """

    # get coordinates if not provided
    if x is None:
        x = np.arange(len(y))

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 6
    else:
        # init_params = copy.deepcopy(init_params)
        init_params = [p for p in init_params]

    # guess reasonable parameters if not provided
    if np.any([ip is None for ip in init_params]):
        to_use = np.logical_not(np.isnan(y))

        bg = np.nanmean(y.ravel())
        A = np.max(y[to_use].ravel()) - bg

        cx, = get_moments(y, order=1, coords=[x])
        m2x, = get_moments(y, order=2, coords=[x])
        sx = np.sqrt(m2x - cx ** 2)

        ip_default = [A, cx, sx, bg, sx, bg]

        # set any parameters that were None to the default values
        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    if bounds is None:
        bounds = ((-np.inf, x.min(), 0, -np.inf, 0, -np.inf),
                  (np.inf, x.max(), x.max() - x.min(), np.inf, x.max() - x.min(), np.inf))

    hg_fn = lambda x, p: (p[0] * np.exp(-(x - p[1])**2 / (2*p[2]**2)) + p[3]) * (x < p[1]) + \
                         ((p[0] + p[3] - p[5]) * np.exp(-(x - p[1])**2 / (2*p[4]**2)) + p[5]) * (x >= p[1])

    result = fit_model(y, lambda p: hg_fn(x, p), init_params, fixed_params=fixed_params, sd=sd, bounds=bounds)

    pfit = result['fit_params']
    fit_fn = lambda x: hg_fn(x, pfit)

    return result, fit_fn

def fit_gauss1d(y, init_params=None, fixed_params=None, sd=None, x=None, bounds=None):
    """
    Fit 1D Gaussian

    :param y:
    :param init_params: [A, cx, sx, bg]
    :param fixed_params:
    :param sd:
    :param x:
    :param bounds:
    :return:
    """

    # get coordinates if not provided
    if x is None:
        x = np.arange(len(y))

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 4
    else:
        init_params = copy.deepcopy(init_params)

    # guess reasonable parameters if not provided
    if np.any([ip is None for ip in init_params]):
        to_use = np.logical_not(np.isnan(y))

        bg = np.nanmean(y.ravel())
        A = np.max(y[to_use].ravel()) - bg

        cx, = get_moments(y, order=1, coords=[x])
        m2x, = get_moments(y, order=2, coords=[x])
        sx = np.sqrt(m2x - cx ** 2)

        ip_default = [A, cx, sx, bg]

        # set any parameters that were None to the default values
        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    if bounds is None:
        bounds = ((-np.inf, x.min(), 0, -np.inf),
                  (np.inf, x.max(), x.max() - x.min(), np.inf))

    fn = lambda p: gauss_fn(x, np.zeros(x.shape), [p[0], p[1], 0, p[2], 1, p[3], 0])
    jacob_fn = lambda p: gauss_jacobian(x, np.zeros(x.shape), [p[0], p[1], 0, p[2], 1, p[3], 0])

    result = fit_model(y, fn, init_params, fixed_params=fixed_params,
                       sd=sd, bounds=bounds, model_jacobian=jacob_fn)

    pfit = result['fit_params']
    fit_fn = lambda x: gauss_fn(x, np.zeros(x.shape), [pfit[0], pfit[1], 0, pfit[2], 1, pfit[3], 0])

    return result, fit_fn

def fit_gauss(img, init_params=None, fixed_params=None, sd=None, xx=None, yy=None, bounds=None):
    """
    Fit 2D gaussian function. The angle theta is defined clockwise from the x- (or y-) axis. NOTE: be careful
    with this when looking at results using e.g. matplotlib.imshow, as this will display the negative y-axis on top.

    :param img: 2D image to fit
    :param init_params: [A, cx, cy, sx, sy, bg, theta]
    :param fixed_params: list of boolean values, same size as init_params.
    :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
    deviation of the mean
    :param xx: 2D array, same size as image (use this instead of 1D array because want to preserve ability to fit on
    non-regularly spaced grids, etc.)
    :param yy:
    :param bounds: (lbs, ubs)
    :return dict results:
    :return fit_fn:
    """

    # get coordinates if not provided
    if xx is None or yy is None:
        xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 7
    else:
        init_params = copy.deepcopy(init_params)

    # guess reasonable parameters if not provided
    if np.any([ip is None for ip in init_params]):
        to_use = np.logical_not(np.isnan(img))

        bg = np.nanmean(img.ravel())
        A = np.max(img[to_use].ravel()) - bg

        cy, cx = get_moments(img, order=1, coords=[yy[:, 0], xx[0, :]])
        m2y, m2x = get_moments(img, order=2, coords=[yy[:, 0], xx[0, :]])
        with np.errstate(invalid='ignore'):
            sx = np.sqrt(m2x - cx ** 2)
            sy = np.sqrt(m2y - cy ** 2)

        ip_default = [A, cx, cy, sx, sy, bg, 0]

        # set any parameters that were None to the default values
        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    if bounds is None:
        bounds = ((-np.inf, xx.min(), yy.min(), 0, 0, -np.inf, -np.inf),
                  (np.inf, xx.max(), yy.max(), xx.max() - xx.min(), yy.max() - yy.min(), np.inf, np.inf))

    result = fit_model(img, lambda p: gauss_fn(xx, yy, p), init_params, fixed_params=fixed_params,
                       sd=sd, bounds=bounds, model_jacobian=lambda p: gauss_jacobian(xx, yy, p))

    pfit = result['fit_params']
    def fit_fn(x, y): return gauss_fn(x, y, pfit)

    return result, fit_fn

def fit_ngauss(img, ngaussians, init_params, fixed_params=None, sd=None, xx=None, yy=None, bounds=None):
    """
    Fit 2D gaussian function. The angle theta is defined clockwise from the x- (or y-) axis. NOTE: be careful
    with this when looking at results using e.g. matplotlib.imshow, as this will display the negative y-axis on top.

    :param img: 2D image to fit
    :param init_params: [A1, cx1, cy1, sx1, sy1, theta1, A2, cx2, ..., thetan, bg]
    :param fixed_params: list of boolean values, same size as init_params.
    :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
    deviation of the mean
    :param xx: 2D array, same size as image (use this instead of 1D array because want to preserve ability to fit on
    non-regularly spaced grids, etc.)
    :param yy:
    :param bounds: (lbs, ubs)
    :return:
    """

    # get coordinates if not provided
    if xx is None or yy is None:
        xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    nparams = 6 * ngaussians + 1
    # get default initial parameters
    if init_params is None:
        init_params = [None] * nparams
    else:
        init_params = copy.deepcopy(init_params)

    if bounds is None:
        bounds = [[-np.inf, xx.min(), yy.min(), 0, 0, -np.inf] * ngaussians + [-np.inf],
                  [ np.inf, xx.max(), yy.max(), xx.max() - xx.min(), yy.max() - yy.min(), np.inf] * ngaussians + [np.inf]]

    result = fit_model(img, lambda p: ngauss_fn(xx, yy, p), init_params, fixed_params=fixed_params,
                       sd=sd, bounds=bounds, model_jacobian=lambda p: ngauss_jacobian(xx, yy, p))

    pfit = result['fit_params']

    def fn(x, y):
        return ngauss_fn(x, y, pfit)

    return result, fn

def fit_model(img, model_fn, init_params, fixed_params=None, sd=None, bounds=None, model_jacobian=None, **kwargs):
    """
    # todo: to be fully general, maybe should get rid of the img argument and only take model_fn. Then this function
    # todo: can be used as a wrapper for least_squares, but adding the ability to fix parameters
    # todo: added function below fit_least_squares(). This function should now call that one
    Fit 2D model function
    :param np.array img: nd array
    :param model_fn: function f(p)
    :param list[float] init_params: p = [p1, p2, ..., pn]
    :param list[boolean] fixed_params: list of boolean values, same size as init_params. If None, no parameters will be fixed.
    :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
    deviation of the mean. If None, then will use a value of 1 for all points. As long as these values are all the same
    they will not affect the optimization results, although they will affect chi squared.
    :param tuple[tuple[float]] bounds: (lbs, ubs). If None, -/+ infinity used for all parameters.
    :param model_jacobian: Jacobian of the model function as a list, [df/dp[0], df/dp[1], ...]. If None, no jacobian used.
    :param kwargs: additional key word arguments will be passed through to scipy.optimize.least_squares
    :return:
    """
    to_use = np.logical_not(np.isnan(img))

    # get default fixed parameters
    if fixed_params is None:
        fixed_params = [False for _ in init_params]

    if sd is None or np.all(np.isnan(sd)) or np.all(sd == 0):
        sd = np.ones(img.shape)

    # handle uncertainties that will cause fitting to fail
    if np.any(sd == 0) or np.any(np.isnan(sd)):
        sd[sd == 0] = np.nanmean(sd[sd != 0])
        sd[np.isnan(sd)] = np.nanmean(sd[sd != 0])

    # default bounds
    if bounds is None:
        bounds = (tuple([-np.inf] * len(init_params)), tuple([np.inf] * len(init_params)))

    # init_params = copy.deepcopy(init_params)
    init_params = np.array(init_params, copy=True)
    # ensure initial parameters within bounds, but don't touch if parameter is fixed
    for ii in range(len(init_params)):
        if (init_params[ii] < bounds[0][ii] or init_params[ii] > bounds[1][ii]) and not fixed_params[ii]:
            if bounds[0][ii] == -np.inf:
                init_params[ii] = bounds[0][ii] + 1
            elif bounds[1][ii] == np.inf:
                init_params[ii] = bounds[1][ii] - 1
            else:
                init_params[ii] = 0.5 * (bounds[0][ii] + bounds[1][ii])

    if np.any(np.isnan(init_params)):
        raise Exception("init_params cannot include nans")

    if np.any(np.isnan(bounds)):
        raise Exception("bounds cannot include nans")

    def err_fn(p): return np.divide(model_fn(p)[to_use].ravel() - img[to_use].ravel(), sd[to_use].ravel())
    if model_jacobian is not None:
        def jac_fn(p): return [v[to_use] / sd[to_use] for v in model_jacobian(p)]

    # if some parameters are fixed, we need to hide them from the fit function to produce correct covariance, etc.
    # awful list comprehension. The idea is this: map the "reduced" (i.e. not fixed) parameters onto the full parameter list.
    # do this by looking at each parameter. If it is supposed to be "fixed" substitute the initial parameter. If not,
    # then get the next value from pfree. We find the right index of pfree by summing the number of previously unfixed parameters
    free_inds = [int(np.sum(np.logical_not(fixed_params[:ii]))) for ii in range(len(fixed_params))]
    def pfree2pfull(pfree): return np.array([pfree[free_inds[ii]] if not fp else init_params[ii] for ii, fp in enumerate(fixed_params)])
    # map full parameters to reduced set
    def pfull2pfree(pfull): return np.array([p for p, fp in zip(pfull, fixed_params) if not fp])

    # function to minimize the sum of squares of, now as a function of only the free parameters
    def err_fn_pfree(pfree): return err_fn(pfree2pfull(pfree))
    if model_jacobian is not None:
        def jac_fn_free(pfree): return pfull2pfree(jac_fn(pfree2pfull(pfree))).transpose()
    init_params_free = pfull2pfree(init_params)
    bounds_free = (tuple(pfull2pfree(bounds[0])), tuple(pfull2pfree(bounds[1])))

    # non-linear least squares fit
    if model_jacobian is None:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free, **kwargs)
    else:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free,
                                                jac=jac_fn_free, x_scale='jac', **kwargs)
    pfit = pfree2pfull(fit_info['x'])

    # calculate chi squared
    nfree_params = np.sum(np.logical_not(fixed_params))
    red_chi_sq = np.sum(np.square(err_fn(pfit))) / (img[to_use].size - nfree_params)

    # calculate covariances
    try:
        jacobian = fit_info['jac']
        cov_free = red_chi_sq * np.linalg.inv(jacobian.transpose().dot(jacobian))
    except np.linalg.LinAlgError:
        cov_free = np.nan * np.zeros((jacobian.shape[1], jacobian.shape[1]))

    cov = np.nan * np.zeros((len(init_params), len(init_params)))
    ii_free = 0
    for ii, fpi in enumerate(fixed_params):
        jj_free = 0
        for jj, fpj in enumerate(fixed_params):
            if not fpi and not fpj:
                cov[ii, jj] = cov_free[ii_free, jj_free]
                jj_free += 1
                if jj_free == nfree_params:
                    ii_free += 1

    result = {'fit_params': pfit, 'chi_squared': red_chi_sq, 'covariance': cov,
              'init_params': init_params, 'fixed_params': fixed_params, 'bounds': bounds,
              'cost': fit_info['cost'], 'optimality': fit_info['optimality'],
              'nfev': fit_info['nfev'], 'njev': fit_info['njev'], 'status': fit_info['status'],
              'success': fit_info['success'], 'message': fit_info['message']}

    return result

def fit_least_squares(model_fn, init_params, fixed_params=None, bounds=None, model_jacobian=None, **kwargs):
    """
    Non-linear least squares fit, but with ability to fix parameters.

    :param model_fn: function f(p)
    :param list[float] init_params: p = [p1, p2, ..., pn]
    :param list[boolean] fixed_params: list of boolean values, same size as init_params. If None, no parameters will be fixed.
    :param tuple[tuple[float]] bounds: (lbs, ubs). If None, -/+ infinity used for all parameters.
    :param model_jacobian: Jacobian of the model function as a list, [df/dp[0], df/dp[1], ...]. If None, no jacobian used.
    :param kwargs: additional key word arguments will be passed through to scipy.optimize.least_squares
    :return:
    """

    # get default fixed parameters
    if fixed_params is None:
        fixed_params = [False for _ in init_params]

    # default bounds
    if bounds is None:
        bounds = (tuple([-np.inf] * len(init_params)), tuple([np.inf] * len(init_params)))

    # init_params = copy.deepcopy(init_params)
    init_params = np.array(init_params, copy=True)
    # ensure initial parameters within bounds, but don't touch if parameter is fixed
    for ii in range(len(init_params)):
        if (init_params[ii] < bounds[0][ii] or init_params[ii] > bounds[1][ii]) and not fixed_params[ii]:
            if bounds[0][ii] == -np.inf:
                init_params[ii] = bounds[0][ii] + 1
            elif bounds[1][ii] == np.inf:
                init_params[ii] = bounds[1][ii] - 1
            else:
                init_params[ii] = 0.5 * (bounds[0][ii] + bounds[1][ii])

    if np.any(np.isnan(init_params)):
        raise Exception("init_params cannot include nans")

    if np.any(np.isnan(bounds)):
        raise Exception("bounds cannot include nans")

    # if some parameters are fixed, we need to hide them from the fit function to produce correct covariance, etc.
    # awful list comprehension. The idea is this: map the "reduced" (i.e. not fixed) parameters onto the full parameter list.
    # do this by looking at each parameter. If it is supposed to be "fixed" substitute the initial parameter. If not,
    # then get the next value from pfree. We find the right index of pfree by summing the number of previously unfixed parameters
    free_inds = [int(np.sum(np.logical_not(fixed_params[:ii]))) for ii in range(len(fixed_params))]

    def pfree2pfull(pfree):
        return np.array([pfree[free_inds[ii]] if not fp else init_params[ii] for ii, fp in enumerate(fixed_params)])

    # map full parameters to reduced set
    def pfull2pfree(pfull):
        return np.array([p for p, fp in zip(pfull, fixed_params) if not fp])

    # function to minimize the sum of squares of, now as a function of only the free parameters
    def err_fn_pfree(pfree):
        return model_fn(pfree2pfull(pfree))

    if model_jacobian is not None:
        def jac_fn_free(pfree): return pfull2pfree(model_jacobian(pfree2pfull(pfree))).transpose()
    init_params_free = pfull2pfree(init_params)
    bounds_free = (tuple(pfull2pfree(bounds[0])), tuple(pfull2pfree(bounds[1])))

    # non-linear least squares fit
    if model_jacobian is None:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free, **kwargs)
    else:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free,
                                                jac=jac_fn_free, x_scale='jac', **kwargs)
    pfit = pfree2pfull(fit_info['x'])

    # calculate chi squared
    nfree_params = np.sum(np.logical_not(fixed_params))
    red_chi_sq = np.sum(np.square(model_fn(pfit))) / (model_fn(init_params).size - nfree_params)

    # calculate covariances
    try:
        jacobian = fit_info['jac']
        cov_free = red_chi_sq * np.linalg.inv(jacobian.transpose().dot(jacobian))
    except np.linalg.LinAlgError:
        cov_free = np.nan * np.zeros((jacobian.shape[1], jacobian.shape[1]))

    cov = np.nan * np.zeros((len(init_params), len(init_params)))
    ii_free = 0
    for ii, fpi in enumerate(fixed_params):
        jj_free = 0
        for jj, fpj in enumerate(fixed_params):
            if not fpi and not fpj:
                cov[ii, jj] = cov_free[ii_free, jj_free]
                jj_free += 1
                if jj_free == nfree_params:
                    ii_free += 1

    result = {'fit_params': pfit, 'chi_squared': red_chi_sq, 'covariance': cov,
              'init_params': init_params, 'fixed_params': fixed_params, 'bounds': bounds,
              'cost': fit_info['cost'], 'optimality': fit_info['optimality'],
              'nfev': fit_info['nfev'], 'njev': fit_info['njev'], 'status': fit_info['status'],
              'success': fit_info['success'], 'message': fit_info['message']}

    return result

# fft tools
def get_fft_frqs(length, dt=1, centered=True, mode='symmetric'):
    """
    Get frequencies associated with FFT, ordered from largest magnitude negative to largest magnitude positive.

    We are always free to take f -> f + n*dt for any integer n, which allows us to transform between the 'symmetric'
    and 'positive' frequency representations.

    If fftshift=False, the natural frequency representation is the positive one, with
    f = [0, ..., L-1]/(dt*L)

    If fftshift=True, the natural frequency representation is the symmetric one
    If x = [0, ..., L-1], then
    for even length sequences, (L-1) = 2*N+1:
    f = [-(N+1), ..., 0, ..., N]/(dt*L)
    and for odd length sequences, (L-1) = 2*N
    f = [    -N, ..., 0, ..., N]/(dt*L)
    i.e. for sequences of even length, we have one more negative frequency than we have positive frequencies.


    :param length: length of sample
    :param dt: spacing between samples
    :param centered: Bool. Controls the order in which fequencies are returned. If true, return
    frequencies in the order corresponding to fftshift(fft(fn)), i.e. with origin in the center of the array.
    If false, origin is at the edge.
    :param mode: 'symmetric' or 'positive'. Controls which frequencies are repoted as postive/negative.
    If 'positive', return positive representation of all frequencies. If 'symmetric', return frequencies larger
    than length//2 as negative
    :return:
    """

    # generate symmetric, fftshifted frequencies
    if np.mod(length, 2) == 0:
        n = int((length - 2) / 2)
        frqs = np.arange(-(n+1), n+1) / length / dt
    else:
        n = int((length - 1) / 2)
        frqs = np.arange(-n, n+1) / length / dt

    # ifftshift if necessary
    if centered:
        pass
    else:
        # convert from origin at center to origin at edge
        frqs = scipy.fft.ifftshift(frqs)

    # shift back to positive if necessary
    if mode == 'symmetric':
        pass
    elif mode == 'positive':
        frqs[frqs < 0] = frqs[frqs < 0] + 1 / dt
    else:
        raise Exception("mode must be 'symmetric' or 'positive', but was '%s'" % mode)

    return frqs

def get_fft_pos(length, dt=1, centered=True, mode='symmetric'):
    """
    Get position coordinates for use with fast fourier transforms (fft's) using one of several different conventions.

    With the default arguments, will return the appropriate coordinates for the idiom
    array_ft = fftshift(fft2(ifftshift(array)))

    We are always free to change the position by a multiple of the overall length, i.e. x -> x + n*L for n an integer.

    if centered=False,
    pos = [0, 1, ..., L-1] * dt

    if centered=True, then for a sequence of length L, we have
    [- ceil( (L-1)/2), ..., 0, ..., floor( (L-1)/2)]
    which is symmetric for L odd, and has one more positive coordinate for L even

    :param length: length of array
    :param dt: spacing between points
    :param centered: controls the order in which frequencies are returned.
    :param mode: "positive" or "symmetric": control which frequencies are reported as positive vs. negative

    :return pos: list of positions
    """

    # symmetric, centered frequencies
    pos = np.arange(-np.ceil(0.5 * (length - 1)), np.floor(0.5 * (length - 1)) + 1)

    if mode == 'symmetric':
        pass
    elif mode == 'positive':
        pos[pos < 0] = pos[pos < 0] + length
    else:
        raise Exception("mode must be 'symmetric' or 'positive', but was '%s'" % mode)

    if centered:
        pass
    else:
        pos = scipy.fft.ifftshift(pos)

    pos = pos * dt

    return pos

def get_spline_fn(x1, x2, y1, y2, dy1, dy2):
    """

    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param dy1:
    :param dy2:
    :return spline_fn:
    :return spline_deriv:
    :return coeffs:
    """
    # s(x) = a * x**3 + b * x**2 + c * x + d
    # vec = mat * [[a], [b], [c], [d]]
    vec = np.array([[y1], [dy1], [y2], [dy2]])
    mat = np.array([[x1**3, x1**2, x1, 1],
                   [3*x1**2, 2*x1, 1, 0],
                   [x2**3, x2**2, x2, 1],
                   [3*x2**2, 2*x2, 1, 0]])
    coeffs = np.linalg.inv(mat).dot(vec)

    fn = lambda x: coeffs[0, 0] * x**3 + coeffs[1, 0] * x ** 2 + coeffs[2, 0] * x + coeffs[3, 0]
    dfn = lambda x: 3 * coeffs[0, 0] * x**2 + 2 * coeffs[1, 0] * x + coeffs[2, 0]
    return fn, dfn, coeffs

def mix_edges(img, wx=0.1, wy=0.1):
    """

    :param img:
    :param wx:
    :param wy:
    :return:
    """
    ny, nx = img.shape
    x, y = np.meshgrid(range(nx), range(ny))
    x = x / x.max()
    y = y / y.max()

    # mix x edges together so that smoothly interpolate from equal mixture at edge to normal image at wx
    fnx, _, _ = get_spline_fn(0, wx, 0.5, 1, 0, 0)
    x_mix = fnx(x) * (x <= wx) + fnx(1 - x) * (x >= (1 - wx)) + 1 * (x > wx) * (x < (1 - wx))
    img_symmetric_x = img * x_mix + np.flip(img, axis=1) * (1 - x_mix)

    # mix y edges together so that smoothly interpolate from equal mixture at edge to normal image at wy
    fny, _, _ = get_spline_fn(0, wy, 0.5, 1, 0, 0)
    y_mix = fny(y) * (y <= wy) + fny(1 - y) * (y >= (1 - wy)) + 1 * (y > wy) * (y < (1 - wy))
    img_symmetric_y = img * y_mix + np.flip(img, axis=0) * (1 - y_mix)

    img_symmetric = 0.5 * (img_symmetric_x + img_symmetric_y)

    return img_symmetric

# translating images
def translate_pix(img, shifts, dx=1, dy=1, mode='wrap', pad_val=0):
    """
    Translate image by given number of pixels with several different boundary conditions. If the shifts are sx, sy,
    then the image will be shifted by sx/dx and sy/dy. If these are not integers, they will be rounded to the closest
    integer.

    i.e. given img(x, y) return img(x + sx, y + sy).

    :param img: image to translate
    :param shifts: distance to translate along each axis [sx, sy]. If these are not integers, then they will be
    rounded to the closest integer value.
    :param dx: size of pixels in x-direction
    :param mode: 'wrap' or 'no-wrap'
    :return:
    """
    # if not isinstance(shifts[0], int) or not isinstance(shifts[1], int):
    #     raise Exception('pix_shifts must be integers.')

    sx = int(np.round(-shifts[0] / dx))
    sy = int(np.round(-shifts[1] / dy))

    img = np.roll(img, sy, axis=0)
    img = np.roll(img, sx, axis=1)

    if mode == 'wrap':
        pass
    elif mode == 'no-wrap':
        mask = np.ones(img.shape)
        ny, nx = img.shape

        if sx >= 0:
            mask[:, :sx] = 0
        else:
            mask[:, nx+sx:] = 0

        if sy >= 0:
            mask[:sy, :] = 0
        else:
            mask[ny+sy:, :] = 0

        img = img * mask + pad_val * (1 - mask)
    else:
        raise Exception("'mode' must be 'wrap' or 'no-wrap' but was '%s'" % mode)

    return img, -sx, -sy

def translate_im(img, shift, dx=1, dy=None):
    """
    Translate img(y,x) to img(y+yo, x+xo).

    e.g. suppose the pixel spacing dx = 0.05 um and we want to shift the image by 0.0366 um,
    then dx = 0.05 and shift = [0, 0.0366]

    :param img: NumPy array, size ny x nx
    :param shift: [yo, xo], in same units as pixels
    :param dx: pixel size of image along x-direction
    :param dy: pixel size of image along y-direction
    :return img_shifted:
    """

    if dy is None:
        dy = dx

    # must use symmetric frequency representation to do correctly!
    # we are using the FT shift theorem to approximate the Nyquist-Whittaker interpolation formula,
    # but we get an extra phase if we don't use the symmetric rep. AND only works perfectly if size odd
    fx = get_fft_frqs(img.shape[1], dt=dx, centered=False, mode='symmetric')
    fy = get_fft_frqs(img.shape[0], dt=dy, centered=False, mode='symmetric')
    fxfx, fyfy = np.meshgrid(fx, fy)

    # 1. ft
    # 2. multiply by exponential factor
    # 3. inverse ft
    exp_factor = np.exp(1j * 2 * np.pi * (shift[0] * fyfy + shift[1] * fxfx))
    img_shifted = fft.fftshift(fft.ifft2(exp_factor * fft.fft2(fft.ifftshift(img))))

    return img_shifted

def translate_ft(img_ft: np.ndarray, shift_frq, dx=1, dy=None, apodization=None) -> np.ndarray:
    """
    Given img_ft(f), return the translated function
    img_ft_shifted(f) = img_ft(f + shift_frq)
    using the FFT shift relationship, img_ft(f + shift_frq) = F[ exp(-2*pi*i * shift_frq * r) * img(r) ]

    This is an approximation to the Whittaker-Shannon interpolation formula which can be performed using only FFT's

    :param img_ft: fourier transform, with frequencies centered using fftshift
    :param shift_frq: [fx, fy]. Frequency in hertz (i.e. angular frequency is k = 2*pi*f)
    :param dx: pixel size (sampling rate) of real space image in x-direction
    :param dy: pixel size (sampling rate) of real space image in y-direction
    :param apodization: apodization function applied (in both k- and real- space)

    :return img_ft_shifted:
    """
    if dy is None:
        dy = dx

    if apodization is None:
        apodization = 1

    # 1. shift frequencies in img_ft so zero frequency is in corner using ifftshift
    # 2. inverse ft
    # 3. multiply by exponential factor
    # 4. take fourier transform, then shift frequencies back using fftshift

    ny, nx = img_ft.shape
    # must use symmetric frequency representation to do correctly!
    # we are using the FT shift theorem to approximate the Whittaker-Shannon interpolation formula,
    # but we get an extra phase if we don't use the symmetric rep. AND only works perfectly if size odd
    x = get_fft_pos(nx, dx, centered=False, mode='symmetric')
    y = get_fft_pos(ny, dy, centered=False, mode='symmetric')

    exp_factor = np.exp(-1j * 2 * np.pi * (shift_frq[0] * x[None, :] + shift_frq[1] * y[:, None]))
    #ifft2(ifftshift(img_ft)) = ifftshift(img)
    img_ft_shifted = fft.fftshift(fft.fft2(apodization * exp_factor * fft.ifft2(fft.ifftshift(img_ft * apodization))))

    return img_ft_shifted

def shannon_whittaker_interp(x, y, dt=1):
    """
    Get function value between sampling points using Shannon-Whittaker interpolation formula.

    :param x: point to find interpolated function
    :param y: function at points n * dt
    :param dt: sampling rate
    :return: y(x)
    """
    ns = np.arange(len(y))
    y_interp = np.zeros(x.shape)
    for ii in range(x.size):
        ind = np.unravel_index(ii, x.shape)
        y_interp[ind] = np.sum(y * sinc(np.pi * (x[ind] - ns * dt) / dt))

    return y_interp

def sinc(x):
    val = np.sin(x) / x
    val[x == 0] = 1
    return val

# plotting tools
def get_cut_profile(img, start_coord, end_coord, width):
    """
    Get data along a 1D line from img

    todo: would like the option to resample along the new coordinates? Otherwise the finite width
    can lead to artifacts
    :param img: 2D numpy array
    :param start_coord: [xstart, ystart], where the upper left pixel of the array is at [0, 0]
    :param end_coord: [xend, yend]
    :param width: width of cut
    :return xcut: coordinate along the cut (in pixels)
    :return cut: values along the cut
    """
    xstart, ystart = start_coord
    xend, yend = end_coord

    angle = np.arctan( (yend - ystart) / (xend - xstart))

    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    xrot = (xx - xstart) * np.cos(angle) + (yy - ystart) * np.sin(angle)
    yrot = (yy - ystart) * np.cos(angle) - (xx - xstart) * np.sin(angle)

    # line goes from (0,0) to (xend - xstart)
    mask = np.ones(img.shape)
    mask[yrot > 0.5 * width] = 0
    mask[yrot < -0.5 * width] = 0
    mask[xrot < 0] = 0
    xrot_end = (xend - xstart) * np.cos(angle) + (yend - ystart) * np.sin(angle)
    mask[xrot > xrot_end] = 0

    xcut = xrot[mask != 0]
    cut = img[mask != 0]

    # sort by coordinate
    inds = np.argsort(xcut)
    xcut = xcut[inds]
    cut = cut[inds]

    return xcut, cut

if __name__ == '__main__':
    pass

