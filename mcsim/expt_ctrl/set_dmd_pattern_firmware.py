"""
Python script which takes command line arguments to program the DMD pattern sequence from patterns previously loaded
into DMD firmware. Firmware loading can be done using the Texas Instruments DLP6500 and DLP9000 GUI
(https://www.ti.com/tool/DLPC900REF-SW)

This file contains information about the DMD firmware patterns in the variables `channel_map` and two helper functions
`get_dmd_sequence()` and `program_dmd_seq()`. This can imported from other python files and called from there,
or this script can be run from the command line.

Run "python set_dmd_pattern_firmware.py -h" on the commandline for a detailed description of command options

Information about which patterns are stored in the firmware and mapping onto different "channels" which
are associated with a particular excitation wavelength and "modes" which are associated with a certain DMD
pattern or set of DMD patterns this are stored in the variable channel_map
This is of type dict{dict{dict{np.array}}}
The top level dictionary keys are the names for each "channel"
The second level down dictionary keys are the names of the "modes" for that channel,
i.e. modes = channel_map["channel_name"].keys()
note that all channels must have a mode called 'default'
The third level dictionary specify the DMD patterns, i.e.
# modes["default"] = {"picture_indices", "bit_indices"}
So e.g. for channel="blue" to get the picture indices associated with the "sim" mode slice as follows:
channel_map["blue"]["sim"]["picture_indices"]
"""

import numpy as np
import argparse
from mcsim.analysis import dmd_patterns
from mcsim.expt_ctrl import dlp6500

dmd_size = np.array([1080, 1920], dtype=int)

# #######################
# define channels and modes
# #######################
channel_map = {"off": {"default": {"picture_indices": np.array([1]), "bit_indices": np.array([4])}},
               "on":  {"default": {"picture_indices": np.array([1]), "bit_indices": np.array([3])}},
               "blue": {"default": {"picture_indices": np.zeros(9, dtype=int), "bit_indices": np.arange(9, dtype=int)}},
               "red": {"default": {"picture_indices": np.zeros(9, dtype=int), "bit_indices": np.arange(9, 18, dtype=int)}},
               "green": {"default": {"picture_indices": np.array([0] * 6 + [1] * 3, dtype=int), "bit_indices": np.array(list(range(18, 24)) + list(range(3)), dtype=int)}},
               "odt": {#"default": {"picture_indices": np.array([1, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13], dtype=int),
                        #            "bit_indices": np.array([7, 18, 23, 4, 9, 14, 19, 0, 5, 10, 15], dtype=int)},
                       "default": {"picture_indices": np.array([1, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13], dtype=int),
                                    "bit_indices": np.array([7, 18, 23, 4, 7, 12, 17, 22, 3, 6, 11], dtype=int)},
                       "n=1_f=0%": {"picture_indices": np.array([1], dtype=int), "bit_indices": np.array([7], dtype=int)},
                       # r = 5
                       # "n=7_f=97%": {"picture_indices": np.array([1, 13, 14, 14, 14, 14, 14], dtype=int),
                       #              "bit_indices": np.array([7, 20, 4, 5, 13, 21, 22], dtype=int)},
                       # "n=11_f=84%": {"picture_indices": np.array([1, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13], dtype=int),
                       #              "bit_indices": np.array([7, 18, 23, 4, 9, 14, 19, 0, 5, 10, 15], dtype=int)},
                       # "n=6_f=84%": {"picture_indices": np.array([1, 11, 12, 12, 13, 13], dtype=int),
                       #              "bit_indices": np.array([7, 18, 4, 14, 0, 10], dtype=int)},
                       # r = 10/15
                       "n=7_f=97%": {"picture_indices": np.array([1, 13, 14, 14, 14, 14, 14], dtype=int),
                                     "bit_indices": np.array([7, 16, 0, 1, 9, 17, 18], dtype=int)},
                       "n=11_f=84%": {"picture_indices": np.array([1, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13], dtype=int),
                                      "bit_indices": np.array([7, 18, 23, 4, 7, 12, 17, 22, 3, 6, 11], dtype=int)},
                       "n=6_f=84%": {"picture_indices": np.array([1, 11, 12, 12, 12, 13], dtype=int),
                                     "bit_indices": np.array([7, 18, 4, 12, 22, 6], dtype=int)},
                       "n=11_f=55%": {"picture_indices": np.array([1, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7], dtype=int),
                                    "bit_indices": np.array([7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9], dtype=int)},
                       "n=6_f=55%": {"picture_indices": np.array([1, 5, 5, 6, 6, 7], dtype=int),
                                    "bit_indices": np.array([7, 12, 22, 8, 18, 4], dtype=int)},
                       "all": {"picture_indices": np.hstack((1 * np.ones([17], dtype=int),
                                                             2 * np.ones([24], dtype=int),
                                                             3 * np.ones([24], dtype=int),
                                                             4 * np.ones([24], dtype=int),
                                                             5 * np.ones([24], dtype=int),
                                                             6 * np.ones([24], dtype=int),
                                                             7 * np.ones([24], dtype=int),
                                                             8 * np.ones([24], dtype=int),
                                                             9 * np.ones([24], dtype=int),
                                                             10 * np.ones([24], dtype=int),
                                                             11 * np.ones([24], dtype=int),
                                                             12 * np.ones([24], dtype=int),
                                                             13 * np.ones([24], dtype=int),
                                                             14 * np.ones([24], dtype=int),
                                                             15 * np.ones([2], dtype=int)
                                                             #15 * np.ones([6], dtype=int),
                                                             )),
                               "bit_indices": np.hstack((np.arange(7, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         np.arange(0, 24, dtype=int),
                                                         #np.arange(0, 6, dtype=int)
                                                         np.arange(0, 2, dtype=int)
                                                         ))}
                       }
            }

# add on/off modes
on_mode = channel_map["on"]["default"]
off_mode = channel_map["off"]["default"]
on_affine_mode = {"picture_indices": np.array([1]), "bit_indices": np.array([5])}
off_affine_mode = {"picture_indices": np.array([1]), "bit_indices": np.array([6])}

# add on/off/widefield/affine/sim modes for other colors
# non-inverted patterns
for m in ["red", "blue"]:
    channel_map[m].update({"off": off_mode})
    channel_map[m].update({"on": on_mode})
    channel_map[m].update({"widefield": on_mode})
    channel_map[m].update({'affine': on_affine_mode})
    channel_map[m].update({'sim': channel_map[m]["default"]})

# inverted-patterns
for m in ["green"]:
    channel_map[m].update({"off": on_mode})
    channel_map[m].update({"on": off_mode})
    channel_map[m].update({"widefield": off_mode})
    channel_map[m].update({'affine': off_affine_mode})
    channel_map[m].update({'sim': channel_map[m]["default"]})

# add angle modes
for m in ["red", "blue", "green"]:
    for angle in range(3):
        pic_inds = channel_map[m]["default"]["picture_indices"][3*angle: 3*(angle+1)]
        bit_inds = channel_map[m]["default"]["bit_indices"][3*angle: 3*(angle+1)]

        channel_map[m].update({f"sim_angle={angle:d}": {"picture_indices": pic_inds,
                                                        "bit_indices": bit_inds}})


# inverted-patterns
for m in ["odt"]:
    channel_map[m].update({"off": on_mode})
    channel_map[m].update({"on": off_mode})


def validate_channel_map(cm):
    """
    check that channel_map is of the correct format
    @param cm:
    @return:
    """
    for ch in list(cm.keys()):
        modes = list(cm[ch].keys())

        if "default" not in modes:
            return False

        for m in modes:
            keys = list(cm[ch][m].keys())
            if "picture_indices" not in keys:
                return False

            pi = cm[ch][m]["picture_indices"]
            if not isinstance(pi, np.ndarray) or pi.ndim != 1:
                return False

            if "bit_indices" not in keys:
                return False

            bi = cm[ch][m]["bit_indices"]
            if not isinstance(bi, np.ndarray) or bi.ndim != 1:
                return False

    return True


# #######################
# firmware patterns
# #######################
def firmware_index_2pic_bit(firmware_indices):
    """
    convert from single firmware pattern index to picture and bit indices
    @param firmware_indices:
    @return:
    """
    pic_inds = firmware_indices // 24
    bit_inds = firmware_indices - 24 * pic_inds

    return pic_inds, bit_inds


def pic_bit_ind_2firmware_ind(pic_inds, bit_inds):
    """
    Convert from picture and bit indices to single firmware pattern index
    @param pic_inds:
    @param bit_inds:
    @return:
    """
    firmware_inds = pic_inds * 24 + bit_inds
    return firmware_inds


def generate_firmware_patterns(generate_patterns=True):
    """
    Generate firmware patterns

    TODO: probably better to add arguments to this and store data in a configuration text file or etc.
    @param generate_patterns:
    @return:
    """
    # ##########################
    # sim patterns
    # ##########################
    nphases = 3
    a1_vecs = np.array([[-3, 11],
                      [-11, 3],
                      [-13, -12],
                      [-5, 18],
                      [-18, 5],
                      [-11, -10],
                      [-3, 11],
                      [-11, 3],
                      [-13, -12]], dtype=int)

    a2_vecs = np.array([[3, 12],
                      [12, 3],
                      [12, 3],
                      [-15, 24],
                      [-24, 15],
                      [15, 3],
                      [3, 15],
                      [15, 3],
                      [3, 12]], dtype=int)

    pattern_inverted = np.array([False, False, False, False, False, False, True, True, True])

    sim_patterns = []
    sim_pattern_data = []
    ny, nx = dmd_size
    for a1, a2, inv in zip(a1_vecs, a2_vecs, pattern_inverted):
        for ii in range(nphases):
            if generate_patterns:
                pattern, _ = dmd_patterns.get_sim_pattern([nx, ny], a1, a2, nphases, phase_index=ii)
                if inv:
                    pattern = 1 - pattern
                sim_patterns.append(pattern)
            sim_pattern_data.append({"type": "sim", "a1": a1, "a2": a2, "index": ii})

    # ######################################
    # odt patterns
    # ######################################
    ang = -45 * np.pi/180
    frq = np.array([np.sin(ang), np.cos(ang)]) * 1/4 * np.sqrt(2)

    # rad = 5
    # rad = 10
    rad = 15
    phase = 0

    # pupil info
    # na_detection = 0.55
    na_detection = 1
    dm = 7.56 # DMD mirror size
    fl_detection = 3e3
    fl_excitation = 1.8e3
    # magnification between DMD and Mitutoyo BFP
    mag_dmd2bfp = 100 / 200 * 300 / 400 * fl_detection / fl_excitation

    pupil_rad_mirrors = fl_detection * na_detection / mag_dmd2bfp / dm

    # patterns
    n_phis = 50
    fractions = [0.25, 0.45, 0.55, 0.65, 0.75, 0.84, 0.95]
    phis = np.arange(n_phis) * 2*np.pi / n_phis
    n_thetas = len(fractions)

    pp, ff = np.meshgrid(phis, fractions)

    xoffs = np.zeros((n_thetas, n_phis))
    yoffs = np.zeros(xoffs.shape)
    for ii in range(n_thetas):
        for jj in range(n_phis):
            xoffs[ii, jj] = np.cos(phis[jj]) * pupil_rad_mirrors * fractions[ii]
            yoffs[ii, jj] = np.sin(phis[jj]) * pupil_rad_mirrors * fractions[ii]

    # remove any points beyond DMD boundary
    in_bounds = np.logical_and.reduce((yoffs + rad < ny//2,
                                       yoffs - rad > -ny//2,
                                       xoffs + rad < nx//2,
                                       xoffs - rad > -nx//2))

    # pp = pp.ravel()
    # ff = ff.ravel()

    ff = np.concatenate((np.array([0]), ff[in_bounds]))
    pp = np.concatenate((np.array([0]), pp[in_bounds]))
    xoffs = np.concatenate((np.array([0]), xoffs[in_bounds].ravel()))
    yoffs = np.concatenate((np.array([0]), yoffs[in_bounds].ravel()))

    if generate_patterns:
        # generate ODT patterns
        cref = np.array([ny // 2, nx // 2])
        xx, yy = np.meshgrid(range(nx), range(ny))

        pattern_base = np.round(np.cos(2 * np.pi * (xx * frq[0] + yy * frq[1]) + phase), 12)
        pattern_base[pattern_base <= 0] = 0
        pattern_base[pattern_base > 0] = 1
        pattern_base = 1 - pattern_base

    npatterns = xoffs.size
    odt_patterns = np.ones((npatterns, ny, nx), dtype=bool)
    odt_pattern_data = []
    for ii in range(npatterns):
        if generate_patterns:
            odt_patterns[ii] = np.copy(pattern_base)
            odt_patterns[ii, np.sqrt((xx - cref[1] - xoffs.ravel()[ii])**2 +
                                     (yy - cref[0] - yoffs.ravel()[ii])**2) > rad] = 1

        odt_pattern_data.append({"type": "odt", "xoffset": xoffs[ii], "yoffset": yoffs[ii],
                                 "angle": ang, "frequency": frq, "phase": phase, "radius": rad, # carrier frequency information
                                 "pupil_frequency_fraction": ff[ii],
                                 "pupil_angle": pp[ii]})

    # ######################################
    # generate other patterns
    # ######################################
    if generate_patterns:
        on_pattern = np.ones((ny, nx), dtype=bool)
        off_pattern = np.zeros((ny, nx), dtype=bool)

        affine_on_pattern, _, _ = dmd_patterns.get_affine_fit_pattern([nx, ny], radii=[3])
        affine_on_pattern = affine_on_pattern.astype(bool).squeeze()
        affine_off_pattern = 1 - affine_on_pattern


        # assemble all patterns
        patterns = sim_patterns + [on_pattern, off_pattern, affine_on_pattern, affine_off_pattern] + [p for p in odt_patterns]
        patterns = np.stack(patterns, axis=0)
    else:
        patterns = None

    pattern_data = sim_pattern_data + \
                   [{"type": "on"}, {"type": "off"}, {"type": "affine on"}, {"type": "affine off"}] + \
                   odt_pattern_data

    npatterns = len(pattern_data)
    npics_needed = int(np.ceil(npatterns / 24))
    bit_inds = np.tile(np.arange(24, dtype=int), [npics_needed])[:npatterns]
    pic_inds = np.concatenate([ii * np.ones(24, dtype=int) for ii in range(npics_needed)])[:npatterns]

    return patterns, pattern_data, pic_inds, bit_inds


def get_dmd_sequence(modes: list[str], channels: list[str], nrepeats: list[int], ndarkframes: int,
                     blank: list[bool], mode_pattern_indices=None):
    """
    Generate DMD patterns from a list of modes and channels
    @param modes:
    @param channels:
    @param nrepeats:
    @param ndarkframes:
    @param blank:
    @param mode_pattern_indices:
    @return picture_indices, bit_indices:
    """
    # check channel argument
    if isinstance(channels, str):
        channels = [channels]

    if not isinstance(channels, list):
        raise ValueError()

    nmodes = len(channels)

    # check mode argument
    if isinstance(modes, str):
        modes = [modes]

    if not isinstance(modes, list):
        raise ValueError()

    if len(modes) == 1 and nmodes > 1:
        modes = modes * nmodes

    if len(modes) != nmodes:
        raise ValueError()

    # check pattern indices argument
    if mode_pattern_indices is None:
        mode_pattern_indices = []
        for c, m in zip(channels, modes):
            npatterns = len(channel_map[c][m]["picture_indices"])
            mode_pattern_indices.append(np.arange(npatterns, dtype=int))

    if isinstance(mode_pattern_indices, int):
        mode_pattern_indices = [mode_pattern_indices]

    if not isinstance(mode_pattern_indices, list):
        raise ValueError()

    if len(mode_pattern_indices) == 1 and nmodes > 1:
        mode_pattern_indices = mode_pattern_indices * nmodes

    if len(mode_pattern_indices) != nmodes:
        raise ValueError()

    # check nrepeats correct type
    if isinstance(nrepeats, int):
        nrepeats = [nrepeats]

    if not isinstance(nrepeats, list):
        raise ValueError()

    if nrepeats is None:
        nrepeats = []
        for _ in zip(channels, modes):
            nrepeats.append(1)

    if len(nrepeats) == 1 and nmodes > 1:
        nrepeats = nrepeats * nmodes

    if len(nrepeats) != nmodes:
        raise ValueError()

    # check blank argument
    if isinstance(blank, bool):
        blank = [blank]

    if not isinstance(blank, list):
        raise ValueError()

    if len(blank) == 1 and nmodes > 1:
        blank = blank * nmodes

    if len(blank) != nmodes:
        raise ValueError()

    # processing
    pic_inds = []
    bit_inds = []
    for c, m, ind, nreps in zip(channels, modes, mode_pattern_indices, nrepeats):
        # need np.array(..., copy=True) to don't get references in arrays
        pi = np.array(np.atleast_1d(channel_map[c][m]["picture_indices"]), copy=True)
        bi = np.array(np.atleast_1d(channel_map[c][m]["bit_indices"]), copy=True)
        # select indices
        pi = pi[ind]
        bi = bi[ind]
        # repeats
        pi = np.hstack([pi] * nreps)
        bi = np.hstack([bi] * nreps)

        pic_inds.append(pi)
        bit_inds.append(bi)

    # insert dark frames
    if ndarkframes != 0:
        for ii in range(nmodes):
            ipic_off = channel_map[channels[ii]]["off"]["picture_indices"]
            ibit_off = channel_map[channels[ii]]["off"]["bit_indices"]

            pic_inds[ii] = np.concatenate((ipic_off * np.ones(ndarkframes, dtype=int), pic_inds[ii]), axis=0).astype(int)
            bit_inds[ii] = np.concatenate((ibit_off * np.ones(ndarkframes, dtype=int), bit_inds[ii]), axis=0).astype(int)

    # insert blanking frames
    for ii in range(nmodes):
        if blank[ii]:
            npatterns = len(pic_inds[ii])
            ipic_off = channel_map[channels[ii]]["off"]["picture_indices"]
            ibit_off = channel_map[channels[ii]]["off"]["bit_indices"]

            ipic_new = np.zeros((2 * npatterns), dtype=int)
            ipic_new[::2] = pic_inds[ii]
            ipic_new[1::2] = ipic_off

            ibit_new = np.zeros((2 * npatterns), dtype=int)
            ibit_new[::2] = bit_inds[ii]
            ibit_new[1::2] = ibit_off

            pic_inds[ii] = ipic_new
            bit_inds[ii] = ibit_new

    pic_inds = np.hstack(pic_inds)
    bit_inds = np.hstack(bit_inds)

    return pic_inds, bit_inds

_, data, _, _ = generate_firmware_patterns(False)
n_firmware_patterns = len(data)
firmware_pattern_map = [data[ii*24: min((ii+1)*24, n_firmware_patterns)] for ii in range(int(np.ceil(n_firmware_patterns / 24)))]

# #######################
# program DMD
# #######################
def program_dmd_seq(dmd: dlp6500.dlp6500, modes: list[str], channels: list[str], nrepeats: list[int], ndarkframes: int,
                    blank: list[bool], mode_pattern_indices: list[int], triggered: bool, verbose: bool = False,
                    exp_time_us: int = 105):
    """
    convenience function for generating DMD pattern and programming DMD

    @param dmd:
    @param modes:
    @param channels:
    @param nrepeats:
    @param ndarkframes:
    @param blank:
    @param mode_pattern_indices:
    @param triggered:
    @param verbose:
    @return:
    """

    pic_inds, bit_inds = get_dmd_sequence(modes, channels, nrepeats, ndarkframes, blank, mode_pattern_indices)
    # #########################################
    # DMD commands
    # #########################################
    dmd.debug = verbose

    dmd.start_stop_sequence('stop')

    # check DMD trigger state
    delay1_us, mode_trig1 = dmd.get_trigger_in1()
    # print('trigger1 delay=%dus' % delay1_us)
    # print('trigger1 mode=%d' % mode_trig1)

    # dmd.set_trigger_in2('rising')
    mode_trig2 = dmd.get_trigger_in2()
    # print("trigger2 mode=%d" % mode_trig2)

    dmd.set_pattern_sequence(pic_inds, bit_inds, exp_time_us, 0, triggered=triggered,
                             clear_pattern_after_trigger=False, bit_depth=1, num_repeats=0, mode='pre-stored')

    if verbose:
        # print pattern info
        print("%d picture indices: " % len(pic_inds), end="")
        print(pic_inds)
        print("%d     bit indices: " % len(bit_inds), end="")
        print(bit_inds)
        print("finished programming DMD")

    return pic_inds, bit_inds

if __name__ == "__main__":

    # #######################
    # define arguments
    # #######################

    parser = argparse.ArgumentParser(description="Set DMD pattern sequence from the command line.")

    # allowed channels
    all_channels = list(channel_map.keys())
    parser.add_argument("channels", type=str, nargs="+", choices=all_channels,
                        help="supply the channels to be used in this acquisition as strings separated by spaces")

    # allowed modes
    modes = list(set([m for c in all_channels for m in list(channel_map[c].keys())]))
    modes_help = "supply the modes to be used with each channel as strings separated by spaces." \
                 "each channel supports its own list of modes.\n"
    for c in all_channels:
        modes_with_parenthesis = ["'%s'" % m for m in list(channel_map[c].keys())]
        modes_help += ("channel '%s' supports: " % c) + ", ".join(modes_with_parenthesis) + ".\n"

    parser.add_argument("-m", "--modes", type=str, nargs=1, choices=modes, default=["default"],
                        help=modes_help)

    # pattern indices
    pattern_indices_help = "Among the patterns specified in the subset specified by `channels` and `modes`," \
                           " only run these indices. For a given channel and mode, allowed indices range from 0 to npatterns - 1." \
                           "This options is most commonly used when only a single channel and mode are provided.\n"
    for c in list(channel_map.keys()):
        for m in list(channel_map[c].keys()):
            pattern_indices_help += "channel '%s` and mode '%s' npatterns = %d.\n" % (c, m, len(channel_map[c][m]["picture_indices"]))

    parser.add_argument("-i", "--pattern_indices", type=int, help=pattern_indices_help)

    parser.add_argument("-r", "--nrepeats", type=int, default=1,
                        help="number of times to repeat the patterns specificed by `channels`, `modes`, and `pattern_indices`")

    # other
    parser.add_argument("-t", "--triggered", action="store_true",
                        help="set DMD to wait for trigger before switching pattern")
    parser.add_argument("-d", "--ndarkframes", type=int, default=0,
                        help="set number of dark frames to be added before each color of SIM/widefield pattern")
    parser.add_argument("-b", "--blank", action="store_true",
                        help="set whether or not to insert OFF patterns between each SIM pattern to blank laser")
    parser.add_argument("-v", "--verbose", action="store_true", help="print more verbose DMD programming information")
    args = parser.parse_args()

    if args.verbose:
        print(args)

    # #########################################
    # load DMD and set pattern
    # #########################################
    dmd = dlp6500.dlp6500win(debug=args.verbose)

    pic_inds, bit_inds = program_dmd_seq(dmd, args.modes, args.channels, args.nrepeats, args.ndarkframes, args.blank,
                                         args.pattern_indices, args.triggered, args.verbose)

