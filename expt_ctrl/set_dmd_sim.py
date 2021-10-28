"""
Python script which takes command line arguments to program the DMD pattern sequence from patterns previously loaded
into DMD firmware. Firmware loading can be done using the Texas Instruments DLP6500 and DLP9000 GUI
(https://www.ti.com/tool/DLPC900REF-SW)

Run "python set_dmd_sim.py -h" on the commandline for a detailed description of command options
"""

import os
import sys
import numpy as np
import argparse

# directory where DMD module is found, relative to this file
instrument_dir = ''

# #######################
# information about which patterns are stored in the firmware
# #######################
# all "on" and all "off" patterns positions in DMD firmware
on_pic_inds = 1
on_bit_inds = 3
off_pic_inds = 1
off_bit_inds = 4

# affine calibration patterns position in firmware
affine_on_pic_inds = 1
affine_on_bit_inds = 5
affine_off_pic_inds = 1
affine_off_bit_inds = 6

# SIM pattern positions as stored in DMD firmware. Load these using the DLP GUI
# sublists give multiple aliases for given color
color_alises = [['473', 'blue'], ['635', 'red'], ['532', 'green'], ['405', 'purple']]
inverted = [False, False, True, False]
nangles = 3
nphases = 3

pic_inds_sim = [np.array([0] * 9),
                np.array([0] * 9),
                np.array([0] * 6 + [1] * 3),
                np.array([1] * 9)]

bit_inds_sim = [np.array(list(range(9))),
                np.array(list(range(9, 18))),
                np.array(list(range(18, 24)) + list(range(3))),
                np.array([3] * 9)]

if __name__ == "__main__":

    # #######################
    # define arguments
    # #######################
    parser = argparse.ArgumentParser(description="Set DMD pattern sequence from the command line.")
    parser.add_argument("colors", type=str, nargs="+", choices=[c for alias in color_alises for c in alias],
                        help="supply the colors to be used in this acquisition as strings separated by spaces")
    parser.add_argument("-t", "--triggered", action="store_true",
                        help="set DMD to wait for trigger before switching pattern")
    parser.add_argument("-d", "--darkframes", type=int, default=0,
                        help="set number of dark frames to be added before each color of SIM/widefield pattern")
    parser.add_argument("-b", "--blank", action="store_true",
                        help="set whether or not to insert OFF patterns between each SIM pattern to blank laser")

    modes = ["sim", "widefield", "affine"]
    parser.add_argument("-m", "--mode", type=str, nargs=1, choices=modes, default=["sim"],
                        help="pass 'sim' to run SIM patterns, 'widefield' to run widefield patterns,"
                        "or 'affine' to run affine calibration patterns.")

    # todo: rename this something more sensible
    parser.add_argument("-wr", "--widefieldrepeats", type=int, default=nangles * nphases,
                        help="number of times to repeat widefield or affine patterns. This simulates the SIM sequence,"
                             " replacing the angle/phase loop with repeats of the same pattern. Note that this argument"
                             " only has an effect if the mode is `widefield` or `affine`")

    parser.add_argument("-s", "--singlepattern", action="store_true",
                        help="Display a single pattern on the DMD. Only a single colors argument should be supplied."
                        "The pattern index is chosen using the --angle and --phase arguments")
    parser.add_argument("-a", "--angle", type=int, choices=list(range(nangles)),
                        help="Angle index of SIM pattern. No effect unless --singlepattern option was passed.")
    parser.add_argument("-p", "--phase", type=int, choices=list(range(nphases)),
                        help="Phase index of SIM pattern. No effect unless --singlepattern option was passed.")
    parser.add_argument("-v", "--verbose", action="store_true", help="print more verbose DMD programming information")
    args = parser.parse_args()

    # ensure arguments are compatible
    if args.singlepattern and len(args.colors) != 1:
        parser.error("--singlepattern was passed, but number of color arguments was not 1")

    if args.singlepattern and (args.angle is None or args.phase is None):
        parser.error("--singlepattern was passed, but either --angle or --phase was not passed.")


    # #######################
    # pattern picture/bit indices
    # list where each element is an array giving the picture/bit indices for all images for SIM imaging with that color
    # #######################
    if args.mode[0] == "widefield":
        # widefield
        # pic_inds = [np.atleast_1d(off_pic_inds) if inv else np.atleast_1d(on_pic_inds) for inv in inverted]
        # bit_inds = [np.atleast_1d(off_bit_inds) if inv else np.atleast_1d(on_bit_inds) for inv in inverted]
        pic_inds = [np.atleast_1d([off_pic_inds] * args.widefieldrepeats) if inv else np.atleast_1d([on_pic_inds] * args.widefieldrepeats) for inv in inverted]
        bit_inds = [np.atleast_1d([off_bit_inds] * args.widefieldrepeats) if inv else np.atleast_1d([on_bit_inds] * args.widefieldrepeats) for inv in inverted]
    elif args.mode[0] == "affine":
        pic_inds = [np.atleast_1d([affine_off_pic_inds] * args.widefieldrepeats) if inv else np.atleast_1d([affine_on_pic_inds] * args.widefieldrepeats) for inv in inverted]
        bit_inds = [np.atleast_1d([affine_off_bit_inds] * args.widefieldrepeats) if inv else np.atleast_1d([affine_on_bit_inds] * args.widefieldrepeats) for inv in inverted]
    elif args.mode[0] == "sim":
        pic_inds = pic_inds_sim
        bit_inds = bit_inds_sim
    else:
        raise ValueError("mode value '%s' was not one of the allowed values." % args.mode[0])

    # #######################
    # parse color argument
    # #######################
    color_arg_inds = np.zeros(len(args.colors), dtype=int)
    for ii, c in enumerate(args.colors):
        color_arg_inds[ii] = int([jj for jj, cc in enumerate(color_alises) if c in cc][0])

    # #######################
    # find patterns to set
    # #######################
    if args.singlepattern:
        current_pic_inds = pic_inds[color_arg_inds[0]][nphases * args.angle + args.phase]
        current_bit_inds = bit_inds[color_arg_inds[0]][nphases * args.angle + args.phase]
    else:
        if not args.triggered:
            current_pic_inds = np.concatenate([pic_inds[pi] for pi in color_arg_inds])
            current_bit_inds = np.concatenate([bit_inds[pi] for pi in color_arg_inds])
        else:
            # OFF frames before start of each color
            current_pic_inds = [np.concatenate((np.array([off_pic_inds] * args.darkframes, dtype=int), pic_inds[pi])) if not inverted[pi] else
                                               np.concatenate((np.array([on_pic_inds] * args.darkframes, dtype=int), pic_inds[pi]))
                                               for pi in color_arg_inds]

            current_bit_inds = [np.concatenate((np.array([off_bit_inds] * args.darkframes, dtype=int), bit_inds[pi])) if not inverted[pi] else
                                               np.concatenate((np.array([on_bit_inds] * args.darkframes, dtype=int), bit_inds[pi]))
                                               for pi in color_arg_inds]

            if args.blank:
                # intersperse pattern frames with off frames
                current_pic_inds = [np.concatenate((ps[:, None], np.array([off_pic_inds] * len(ps), dtype=int)[:, None]), axis=1).ravel() if not inverted[pi] else
                                np.concatenate((ps[:, None], np.array([on_pic_inds] * len(ps), dtype=int)[:, None]), axis=1).ravel()
                                for pi, ps in zip(inverted, current_pic_inds)]
                current_bit_inds = [np.concatenate((ps[:, None], np.array([off_bit_inds] * len(ps), dtype=int)[:, None]), axis=1).ravel() if not inverted[pi] else
                                np.concatenate((ps[:, None], np.array([on_bit_inds] * len(ps), dtype=int)[:, None]), axis=1).ravel()
                                for pi, ps in enumerate(current_bit_inds)]

            current_pic_inds = list(np.concatenate(current_pic_inds))
            current_bit_inds = list(np.concatenate(current_bit_inds))

    # #########################################
    # print settings
    # #########################################
    if args.verbose:
        print(args)

    if not args.singlepattern:
        print("%d picture indices: " % len(current_pic_inds), end="")
    print("picture indices: ", end="")
    print(current_pic_inds)
    if not args.singlepattern:
        print("%d    bit indices: " % len(current_bit_inds), end="")
    print("bit indices: ", end="")
    print(current_bit_inds)

    # #########################################
    # DMD commands
    # #########################################
    # add instrument directory to path
    fpath = os.path.realpath(__file__)
    fdir = os.path.dirname(fpath)
    instrument_dir = os.path.join(fdir, instrument_dir)
    sys.path.append(instrument_dir)
    import dlp6500

    # load DMD and set pattern
    dmd = dlp6500.dlp6500win(debug=args.verbose)

    dmd.start_stop_sequence('stop')

    # check DMD trigger state
    delay1_us, mode_trig1 = dmd.get_trigger_in1()
    print('trigger1 delay=%dus' % delay1_us)
    print('trigger1 mode=%d' % mode_trig1)

    # dmd.set_trigger_in2('rising')
    mode_trig2 = dmd.get_trigger_in2()
    print("trigger2 mode=%d" % mode_trig2)

    dmd.set_pattern_sequence(current_pic_inds, current_bit_inds, 105, 0, triggered=args.triggered,
                             clear_pattern_after_trigger=False, bit_depth=1, num_repeats=0, mode='pre-stored')

    print("finished programming DMD")