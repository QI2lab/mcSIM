import datetime
import glob
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile


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
        raise FileExistsError("Path '%s' does not exists." % metadata_dir)

    # todo: are there cases where there are multiple metadata files for one dataset?
    metadata_paths = list(Path(metadata_dir).glob('**/' + file_pattern))
    metadata_paths = sorted(metadata_paths)

    if metadata_paths == []:
        raise FileExistsError("No metadata files matching pattern '%s' found." % file_pattern)

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


def read_mm_dataset(md, time_indices=None, channel_indices=None, z_indices=None, xy_indices=None, user_indices={}):
    """
    Load a set of images from MicroManager metadata, read using parse_mm_metadata()

    :param md: metadata Pandas datable, as created by parse_mm_metadata()
    :param time_indices:
    :param channel_indices:
    :param z_indices:
    :param xy_indices:
    :param user_indices: {"name": indices}
    :return imgs:
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
    imgs = read_multi_tiff(fnames, slices)

    return imgs


def read_tiff(fname, slices=None):
    """
    Read tiff file containing multiple images

    # todo: PIL uses int32 for memory mapping on windows, so cannot address parts of TIF file after 2GB
    # todo: stack overflow https://stackoverflow.com/questions/59485047/importing-large-multilayer-tiff-as-numpy-array
    # todo: pull request https://github.com/python-pillow/Pillow/pull/4310
    # todo: currently PYPI does not have this fix, but Christoph Goehlke version does
    # todo: see here: https://www.lfd.uci.edu/~gohlke/pythonlibs/

    :param fname: path to file
    :param slices: list of slices to read

    :return imgs: 3D numpy array, nz x ny x nx
    :return tiff_metadata: metadata tags with recognized numbers, corresponding to known descriptions
    :return other_metadata: other metadata tags with unrecognized numbers
    """

    tif = tifffile.TiffFile(fname)
    n_frames = len(tif.pages)

    if isinstance(slices, int) or np.issubdtype(type(slices), np.integer):
        slices = [slices]
    if slices is None:
        slices = range(n_frames)

    # read metadata
    tiff_metadata = {}
    for tag in tif.pages[0].tags:
        tiff_metadata[tag.name] = tag.value

    # read
    imgs = tifffile.imread(fname, key=slices)

    if imgs.ndim == 2:
        imgs = np.expand_dims(imgs, axis=0)

    return imgs, tiff_metadata


def read_multi_tiff(fnames, slice_indices):
    """
    Load multiple images and slices, defined by lists fnames and slice_indices,
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
        # todo: replace with tifffile
        imgs_curr, _ = read_tiff(fu, slices)

        # this is necessary in case e.g. one file has images [1,3,7] and another has [2,6,10]
        for jj, ind in enumerate(inds[inds_to_unique == ii]):
            imgs[ind] = imgs_curr[jj, :, :]

    # if any([isinstance(im, str) for im in imgs]):
    #     raise ValueError()
    imgs = np.asarray(imgs)

    return imgs


def save_tiff(img, save_fname, dtype, tif_metadata=None, axes_order='ZYX', hyperstack=False, **kwargs):
    """
    Save an nD NumPy array to a tiff file

    # todo: currently not saving metadata

    :param img: nd-numpy array to save as TIF file
    :param save_fname: path to save file
    :param np.dtype dtype: data type to save tif as
    :param tif_metadata: dictionary of tif metadata. All tags must be recognized.
    :param other_metadata: dictionary of tif metadata with tags that will not be recognized.
    :param axes_order: a string consisting of XYZCTZ where the fastest axes are more to the right
    :param hyperstack: whether or not to save in format compatible with imagej hyperstack
    :return:
    """

    if hyperstack:
        img = tifffile.transpose_axes(img, axes_order, asaxes='TZCYXS')

    tifffile.imwrite(save_fname, img.astype(dtype), dtype=dtype, imagej=True, **kwargs)


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


def tiffs2stack(fname_out, dir_path, fname_exp="*.tif",
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
        raise NotImplementedError("Not implemented for nxy != 1")

    # read image to get image size
    im_first, _ = read_tiff(files[0])
    _, ny, nx = im_first.shape

    # combine images to one array
    imgs = np.zeros((nc, nt, nz, ny, nx)) * np.nan
    for ii in range(len(files)):
        img, _ = read_tiff(files[ii])
        imgs[channels[ii], times[ii], slices[ii]] = img[0]

    if np.any(np.isnan(imgs)):
        print("WARNING: not all channels/times/slices/positions were accounted for. Those that were not found are replaced by NaNs")

    # save results
    img = tifffile.transpose_axes(img, "CTZYX", asaxes='TZCYXS')
    tifffile.imwrite(fname_out, img.astype(np.float32), imagej=True)


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
        raise ValueError("'mode' must be 'file' or 'dir', but was '%s'" % mode)

    ii = 1
    while os.path.exists(fname):
        fname = '%s_%d%s' % (fname_root, ii, ext)
        ii += 1

    return fname


def get_timestamp():
    now = datetime.datetime.now()
    return '%04d_%02d_%02d_%02d;%02d;%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)