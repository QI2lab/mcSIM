"""
Create DAQ program for SIM/ODT experiment

Relies on DAQ line mapping scheme used in daq.py and daq_map.py
"""
from typing import Optional
from collections.abc import Sequence
from re import match
import numpy as np


def get_sim_odt_sequence(daq_do_map: dict,
                         daq_ao_map: dict,
                         presets: dict,
                         acquisition_modes: Sequence[dict],
                         odt_exposure_time: float,
                         sim_exposure_time: float,
                         dt: float,
                         interval: float = 0,
                         n_odt_per_sim: int = 1,
                         n_trig_width: int = 1,
                         dmd_delay: float = 105e-6,
                         odt_stabilize_t: float = 0.,
                         min_odt_frame_time: float = 8e-3,
                         sim_readout_time: float = 10e-3,
                         sim_stabilize_t: float = 200e-3,
                         shutter_delay_time: float = 50e-3,
                         stage_delay_time: float = 0.,
                         n_xy_positions: int = 1,
                         n_times_fast: int = 1,
                         z_voltages: Sequence[float] = None,
                         use_dmd_as_odt_shutter: bool = False,
                         n_digital_ch: int = 16,
                         n_analog_ch: int = 4,
                         parameter_scan: dict = None,
                         turn_lasers_off_interval: bool = False):
    """
    Create DAQ program for SIM/ODT experiment. Support looping in order (from fastest to slowest)
    pattern, channel, z-position, analog parameter, times (fast), position, times (slow)
    This program does not know anything about the times (slow) axis

    :param daq_do_map: e.g. from daq_map.py
    :param daq_ao_map: e.g. from daq_map.py
    :param presets: dictionary of preset channels
    :param acquisition_modes: list of dictionary. Each dictionary contains the keys "channel", "patterns",
       "pattern_mode", "camera", and "npatterns", and "dmd_on_time". Allowed values of "channel" are the keys
       of presets. Allowed values of "patterns" are the keys of presets[channel] (these are not used here).
       Allowed values of "pattern_mode" are "default" or "average". Allowed values of "camera" are
       "cam1", "cam2" or "both". Values of "npatterns" give the number of images associated with the pattern.
    :param odt_exposure_time: odt exposure time in s
    :param sim_exposure_time: sim exposure time in s
    :param dt: daq time step
    :param interval: interval applied after channel loop. This should not be used for timelapse imaging. Instead,
      use DAQ pause trigger.
    :param n_odt_per_sim: number of ODT images to take per each SIM image set
    :param n_trig_width: width of triggers
    :param dmd_delay:
    :param odt_stabilize_t:
    :param min_odt_frame_time:
    :param sim_readout_time:
    :param sim_stabilize_t:
    :param shutter_delay_time:
    :param stage_delay_time: Delay between xy position loops. This will be placed at the start of the loop
    :param n_xy_positions: number of xy-position loops. Actually these are simply duplicates of
    :param n_times_fast: number of time repeats inside position loop.
    :param z_voltages:
    :param bool use_dmd_as_odt_shutter:
    :param n_digital_ch:
    :param n_analog_ch:
    :param parameter_scan: dictionary defining parameter scan. These values will overwrite the values set in 'channels'
    :param turn_lasers_off_interval:
    :return digital_pgm_full, analog_pgm_full, info:
    """

    if z_voltages is None:
        z_voltages = [0]

    # programs for each channel
    digital_pgms = []
    analog_pgms = []
    info = ""
    for ii, am in enumerate(acquisition_modes):

        # assign interval only to last mode
        if ii == len(acquisition_modes) - 1:
            interval_now = interval
        else:
            interval_now = 0

        # current exposure time
        if am["camera"] == "cam1":
            exposure_now = sim_exposure_time
        elif am["camera"] == "cam2":
            exposure_now = odt_exposure_time
        elif am["camera"] == "both":
            exposure_now = np.max([sim_exposure_time, odt_exposure_time])
        else:
            raise ValueError(f"unexpected value for camera given: {am['camera']:s}")

        if am["channel"] == "odt":
            d, a, i = get_odt_sequence(daq_do_map,
                                       daq_ao_map,
                                       presets[am["channel"]],
                                       exposure_now,
                                       am["npatterns"],
                                       dt=dt,
                                       interval=interval_now,
                                       nrepeats=n_odt_per_sim,
                                       n_trig_width=n_trig_width,
                                       dmd_delay=dmd_delay,
                                       stabilize_t=odt_stabilize_t,
                                       min_frame_time=min_odt_frame_time,
                                       shutter_delay_time=shutter_delay_time,
                                       n_digital_ch=n_digital_ch,
                                       n_analog_ch=n_analog_ch,
                                       camera=am["camera"],
                                       dmd=am["dmd"],
                                       average_patterns=am["pattern_mode"] == "average",
                                       use_dmd_as_shutter=use_dmd_as_odt_shutter,
                                       dmd_on_time=am["dmd_on_time"])

            info += i
            digital_pgms.append(d)
            analog_pgms.append(a)
        else:
            d, a, i = get_sim_sequence(daq_do_map,
                                       daq_ao_map,
                                       presets[am["channel"]],
                                       exposure_now,
                                       am["npatterns"],
                                       dt=dt,
                                       interval=interval_now,
                                       nrepeats=1,
                                       n_trig_width=n_trig_width,
                                       dmd_delay=dmd_delay,
                                       stabilize_t=sim_stabilize_t,
                                       min_frame_time=0,
                                       cam_readout_time=sim_readout_time,
                                       shutter_delay_time=shutter_delay_time,
                                       n_digital_ch=n_digital_ch,
                                       n_analog_ch=n_analog_ch,
                                       use_dmd_as_shutter=am["pattern_mode"] != "from-file",
                                       average_patterns=am["pattern_mode"] == "average",
                                       camera=am["camera"],
                                       dmd=am["dmd"],
                                       dmd_on_time=am["dmd_on_time"])

            # if there is only one mode, keep SIM shutter open
            if len(acquisition_modes) == 1:
                d[:, daq_do_map["sim_shutter"]] = 1

            info += i
            digital_pgms.append(d)
            analog_pgms.append(a)

    # programs for each z/parameter scanned
    digital_pgm_one_z = np.vstack(digital_pgms)
    analog_pgms_one_z = np.vstack(analog_pgms)

    # check correct number of analog program steps and analog triggers
    if not analog_pgms_one_z.shape[0] == np.sum(digital_pgm_one_z[:, daq_do_map["analog_trigger"]]):
        raise AssertionError(f"size of analog program="
                             f"{analog_pgms_one_z.shape[0]:d}"
                             f" should equal number of analog triggers="
                             f"{np.sum(digital_pgm_one_z[:, daq_do_map['analog_trigger']]):d}")

    # #######################
    # z-stack logic
    # #######################
    # digital pgm are just repeated. We don't need to do anything
    # analog pgms must be repeated with correct z-voltages
    # get correct voltage for each step
    analog_pgms_per_z = []
    nz = len(z_voltages)
    for v in z_voltages:
        pgm_temp = np.array(analog_pgms_one_z, copy=True)
        pgm_temp[:, daq_ao_map["z_stage"]] = v
        analog_pgms_per_z.append(pgm_temp)

    analog_pgm_full = np.vstack(analog_pgms_per_z)
    analog_pgm_full[:, daq_ao_map["z_stage_monitor"]] = analog_pgm_full[:, daq_ao_map["z_stage"]]

    # #######################
    # parameter_scan
    # #######################
    if parameter_scan is not None:
        scan_lines, scan_volts = zip(*parameter_scan.items())
        nparams = len(scan_volts[0])

        # check all are the same size
        if not np.all(np.array([len(p) == nparams for p in scan_volts])):
            raise ValueError()

        analog_pgms_per_parameter = []
        for ii in range(nparams):
            pgm_temp = np.array(analog_pgm_full, copy=True)

            for jj, line in enumerate(scan_lines):
                pgm_temp[:, daq_ao_map[line]] = scan_volts[jj][ii]
            analog_pgms_per_parameter.append(pgm_temp)

        analog_pgm_full = np.vstack(analog_pgms_per_parameter)
    else:
        nparams = 1

    # #######################
    # logic for each xy-position
    # #######################
    digital_pgm_xy = np.concatenate([digital_pgm_one_z] * nz * nparams * n_times_fast, axis=0)

    n_wait = int(np.ceil(stage_delay_time / dt))
    # duplicate last time-step for wait
    wait_pgm = np.tile(digital_pgm_xy[0][np.newaxis, :], [n_wait, 1])

    # #######################
    # digital program for all xy-positions
    # #######################
    digital_pgm_xy_with_wait = np.concatenate((wait_pgm, digital_pgm_xy), axis=0)
    digital_pgm_full = np.concatenate([digital_pgm_xy_with_wait] * n_xy_positions, axis=0)

    if turn_lasers_off_interval:
        digital_pgm_full[-1, daq_do_map["red_laser"]] = 0
        digital_pgm_full[-1, daq_do_map["blue_laser"]] = 0
        digital_pgm_full[-1, daq_do_map["green_laser"]] = 0

    # #######################
    # print information
    # #######################
    info += "\n".join([repr(am) for am in acquisition_modes]) + "\n"
    info += f"full digital program = {digital_pgm_full.shape[0] * dt * 1e3: 0.3f}ms =" \
            f" {digital_pgm_full.shape[0]:d} clock cycles\n"
    info += f"full analog program = {analog_pgm_full.shape[0]} steps"

    return digital_pgm_full, analog_pgm_full, info


def get_odt_sequence(daq_do_map: dict,
                     daq_ao_map: dict,
                     preset: dict,
                     exposure_time: float,
                     npatterns: int,
                     dt: float,
                     interval: float = 0,
                     nrepeats: int = 1,
                     n_trig_width: int = 1,
                     dmd_delay: float = 105e-6,
                     stabilize_t: float = 0.,
                     min_frame_time: float = 8e-3,
                     shutter_delay_time: float = 50e-3,
                     n_digital_ch: int = 16,
                     n_analog_ch: int = 4,
                     average_patterns: bool = False,
                     camera: str = "cam2",
                     dmd: int = 0,
                     use_dmd_as_shutter: bool = False,
                     dmd_on_time: Optional[float] = None):
    """
    Create DAQ sequence for running optical diffraction tomography experiment. All times given in seconds.

    :param daq_do_map: dictionary with named lines as keys and line numbers as values. Must include lines ...
    :param daq_ao_map: dictionary with named lines as keys and line number as values.
    :param preset: preset used to set analog information. Dictionary with keys "analog" and "digital", where values
    are subentries. In this case, digital entries are ignored.
    :param exposure_time:
    :param npatterns: number of patterns
    :param dt: sampling time step
    :param interval:
    :param nrepeats: number of repeats. Typically used for time/z-stacks
    :param n_trig_width:
    :param dmd_delay: time to trigger DMD before triggering camera
    :param stabilize_t: time to wait for laser to stabilize at the start of sequence
    :param min_frame_time: minimum time before triggering the camera again
    :param shutter_delay_time: delay time to allow shutter to open
    :param n_digital_ch: number of digital lines
    :param n_analog_ch: number of analog lines
    :param average_patterns:
    :param camera:
    :param dmd: specify index of DMD device to use
    :param use_dmd_as_shutter: whether to assume there are 2x as many DMD patterns with "off" pattern interleaved
      between on patterns
    :param dmd_on_time:
    :return: do_odt, ao_odt, info
    """

    info = ""

    if average_patterns:
        raise NotImplementedError("averaging patterns not implemented with ODT")

    if dmd_on_time:
        raise NotImplementedError("using dmd_on_time not implemented with ODT")

    # todo: probably better to pass which lines I want coded to DMD rather than this hokey method
    if dmd == 0:
        dmd_advance = daq_do_map["dmd_advance"]
        dmd_enable = daq_do_map["dmd_enable"]
    elif dmd == 1:
        dmd_advance = daq_do_map["dmd2_advance"]
        dmd_enable = daq_do_map["dmd2_enable"]
    else:
        raise ValueError(f"dmd value must be 0 or 1, but was {dmd}")

    # #########################
    # calculate number of clock cycles for different pieces of sequence
    # #########################
    # number of steps for dmd pre-trigger
    n_dmd_pre_trigger = int(np.round(dmd_delay / dt))
    # delay between frames
    nsteps_interval = int(np.ceil(interval / dt))
    # minimum frame time
    nsteps_min_frame = int(np.ceil(min_frame_time / dt))

    if n_trig_width >= nsteps_min_frame:
        raise ValueError(f"n_trig_width={n_trig_width:d} cannot be longer than nsteps_min_frame={nsteps_min_frame:d}")

    if nsteps_min_frame == 1 and use_dmd_as_shutter:
        raise ValueError("nsteps_min_frame must be > 1 if use_dmd_as_odt_shutter=True")

    # exposure time and frame time
    nsteps_exposure = int(np.ceil(exposure_time / dt))
    nsteps_frame = np.max([nsteps_exposure, nsteps_min_frame])

    # time for laser power to stabilize
    n_odt_stabilize = int(np.ceil(stabilize_t / dt))
    if n_odt_stabilize < n_dmd_pre_trigger:
        n_odt_stabilize = n_dmd_pre_trigger
        info += f"n_odt_stabilize was less than n_dmd_pre_trigger. Increased it to {n_odt_stabilize:d} steps\n"

    # shutter delay
    n_odt_shutter_delay = int(np.ceil(shutter_delay_time / dt))
    if n_odt_stabilize - n_odt_shutter_delay < 0:
        n_odt_shutter_delay = 0

    # active steps
    nsteps_active = nsteps_frame * npatterns * nrepeats + n_odt_stabilize
    # steps including delay interval
    nsteps = np.max([nsteps_active, nsteps_interval])

    # #########################
    # create sequence
    # #########################
    do_odt = np.zeros((nsteps, n_digital_ch), dtype=np.uint8)

    # trigger analog lines to start
    do_odt[0, daq_do_map["analog_trigger"]] = 1
    # shutter on one delay time before imaging
    do_odt[n_odt_stabilize - n_odt_shutter_delay:, daq_do_map["odt_shutter"]] = 1
    # laser always on
    do_odt[:, daq_do_map["odt_laser"]] = 1
    # DMD enable trigger always on
    do_odt[:, dmd_enable] = 1

    if camera in ["cam2", "both"]:
        # camera enable trigger (must be high for advance trigger)
        do_odt[:, daq_do_map["odt_cam_enable"]] = 1  # photron/phantom camera

        # set camera trigger, which starts after delay time for DMD to display pattern
        do_odt[n_odt_stabilize:nsteps_active:nsteps_frame, daq_do_map["odt_cam_sync"]] = 1
        # for debugging, make trigger pulse longer to see on scope
        for ii in range(n_trig_width):
            do_odt[n_odt_stabilize + ii:nsteps_active:nsteps_frame, daq_do_map["odt_cam_sync"]] = 1

    if camera in ["cam1", "both"]:
        # camera enable trigger (must be high for advance trigger)
        # do_odt[:, daq_do_map["odt_cam_enable"]] = 1  # photron/phantom camera

        # set camera trigger, which starts after delay time for DMD to display pattern
        do_odt[n_odt_stabilize:nsteps_active:nsteps_frame, daq_do_map["sim_cam_sync"]] = 1
        # for debugging, make trigger pulse longer to see on scope
        for ii in range(n_trig_width):
            do_odt[n_odt_stabilize + ii:nsteps_active:nsteps_frame, daq_do_map["sim_cam_sync"]] = 1

    # DMD advance trigger
    do_odt[n_odt_stabilize - n_dmd_pre_trigger:nsteps_active-nsteps_frame:nsteps_frame, dmd_advance] = 1

    # ending point is -nsteps_odt_frame to avoid having extra trigger at the end
    # which is really the pretrigger for the next frame
    if use_dmd_as_shutter:
        # extra advance trigger to "turn off" DMD and end exposure
        do_odt[n_odt_stabilize - n_dmd_pre_trigger + nsteps_exposure:nsteps_active:nsteps_frame,
               dmd_advance] = 1

    # monitor lines
    do_odt[:, daq_do_map["signal_monitor"]] = do_odt[:, dmd_advance]
    do_odt[:, daq_do_map["camera_trigger_monitor"]] = np.logical_or(do_odt[:, daq_do_map["sim_cam_sync"]],
                                                                    do_odt[:, daq_do_map["odt_cam_sync"]])

    # set analog channels to match given preset
    ao_odt = np.zeros((1, n_analog_ch))
    for k in preset["analog"].keys():
        ao_odt[:, daq_ao_map[k]] = preset["analog"][k]

    # useful information to print
    info += f"odt stabilize time = {stabilize_t * 1e3:.3f}ms = {n_odt_stabilize:d} clock cycles\n"
    info += f"odt exposure time = {exposure_time * 1e3:.3f}ms = {nsteps_exposure:d} clock cycles\n"
    info += f"odt one frame = {nsteps_frame * dt * 1e3:.3f}ms = {nsteps_frame:d} clock cycles\n"
    info += f"odt one sequence of {nrepeats:d} volumes = " \
            f"{nsteps_active * dt * 1e3:.3f}ms = " \
            f"{nsteps_active:d} clock cycles\n"

    return do_odt, ao_odt, info


def get_sim_sequence(daq_do_map: dict,
                     daq_ao_map: dict,
                     preset: dict,
                     exposure_time: float,
                     npatterns: int,
                     dt: float,
                     interval: float = 0.,
                     nrepeats: int = 1,
                     n_trig_width: int = 1,
                     dmd_delay: float = 105e-6,
                     stabilize_t: float = 200e-3,
                     min_frame_time: float = 0.,
                     cam_readout_time: float = 10e-3,
                     shutter_delay_time: float = 50e-3,
                     n_digital_ch: int = 16,
                     n_analog_ch: int = 4,
                     use_dmd_as_shutter: bool = True,
                     average_patterns: bool = False,
                     camera: str = "cam1",
                     dmd: int = 0,
                     force_equal_subpatterns: bool = True,
                     turn_laser_off_during_interval: bool = True,
                     dmd_on_time: Optional[float] = None):
    """
    Generate DAQ array for running a SIM experiment. Also supports taking a pseudo-widefield image by
    running through all SIM patterns during one camera exposure

    :param daq_do_map:
    :param daq_ao_map:
    :param preset:
    :param exposure_time: camera exposure time
    :param npatterns:
    :param dt:
    :param interval: interval between camera images
    :param nrepeats:
    :param n_trig_width: DMD trigger width signal in # of timesteps
    :param dmd_delay: DMD should be pre-triggered by this amount to come on when desired
    :param stabilize_t:
    :param min_frame_time:
    :param cam_readout_time:
    :param shutter_delay_time:
    :param n_digital_ch:
    :param n_analog_ch:
    :param use_dmd_as_shutter: if True, add extra trigger signal to advance DMD pattern to "OFF" state. In non-average
      mode there is on display-pattern and on off-pattern per frame, and these alternate. In average mode there are
      npatterns display-patterns followed by one off-pattern per frame.
    :param average_patterns: display all patterns sequentially in frame
    :param camera: "cam1", "cam2", or "both"
    :param dmd: specify which DMD device to use
    :param force_equal_subpatterns:
    :param turn_laser_off_during_interval:
    :param dmd_on_time: time DMD is on
    :return do_channel, ao_channel, info:
    """

    info = ""

    if nrepeats != 1:
        raise NotImplementedError("only nrepeats=1 is implemented")

    if min_frame_time != 0:
        raise NotImplementedError("only min_frame_time=0 is implemented")

    # todo: probably better to pass which lines I want coded to DMD rather than this hokey method
    if dmd == 0:
        dmd_advance = daq_do_map["dmd_advance"]
        dmd_enable = daq_do_map["dmd_enable"]
    elif dmd == 1:
        dmd_advance = daq_do_map["dmd2_advance"]
        dmd_enable = daq_do_map["dmd2_enable"]
    else:
        raise ValueError(f"dmd value must be 0 or 1, but was {dmd}")

    # #################################
    # total frame time
    # #################################
    # DMD pre-trigger time
    n_dmd_pre_trigger = int(np.round(dmd_delay / dt))

    # time to stabilize (laser power/mirror)
    n_stabilize = int(np.ceil(stabilize_t / dt))

    n_sim_shutter_delay = int(np.ceil(shutter_delay_time / dt))
    if n_stabilize - n_sim_shutter_delay < 0:
        n_sim_shutter_delay = 0

    # delay between frames
    n_interval = int(np.ceil(interval / dt))

    # time for camera to roll open/closed
    # todo: probably prefer to take ceil(), but shouldn't matter much a long as dt is much smaller than exposure
    n_readout = int(np.round(cam_readout_time / dt))

    # camera exposure time
    n_cam_exposure = int(np.ceil(exposure_time / dt))
    n_cam_all_pixels_exposure = n_cam_exposure - 2 * n_readout

    if n_cam_all_pixels_exposure < n_dmd_pre_trigger:
        raise ValueError(f"Number of timesteps where all pixels are being exposed"
                         f" is less than DMD pre-trigger timesteps {n_cam_all_pixels_exposure:d}<{n_dmd_pre_trigger:d}")

    # todo: can this be less ... or is this the limit due to hardware triggering?
    # frame time
    n_frame = n_cam_exposure + 2 * n_readout

    # #################################
    # pattern times
    # #################################

    # if using "average" mode, all patterns displayed during one frame time
    # note that these numbers do not account for off-frames used to shutter DMD
    if average_patterns:
        n_display_patterns_frame = npatterns
        npatterns = 1
    else:
        n_display_patterns_frame = 1

    # total number of patterns, including off patterns
    if use_dmd_as_shutter:
        n_total_patterns_frame = n_display_patterns_frame + 1
    else:
        n_total_patterns_frame = n_display_patterns_frame

    if average_patterns and not use_dmd_as_shutter:
        raise NotImplementedError("currently average_patterns required use_dmd_as_shutter also be selected")

    # increase camera exposure time to cover an integer number of sub-frame patterns
    if average_patterns and force_equal_subpatterns and dmd_on_time is None:
        n_cam_exposure += n_cam_exposure % n_total_patterns_frame

    # time to keep DMD on for each sub-frame pattern
    if dmd_on_time is not None:
        if not use_dmd_as_shutter:
            raise ValueError("a value for dmd_on_time was provided, but the DMD is not being run as a shutter,"
                             "so this cannot be implemented.")

        # if using more than one pattern, have to correct for this
        n_dmd_on = int(np.ceil(dmd_on_time / n_display_patterns_frame / dt))
        n_dmd_off = n_cam_all_pixels_exposure - n_dmd_on * n_display_patterns_frame
    else:
        # n_dmd_on = int(np.floor(n_cam_exposure / (n_display_patterns_frame + 1)))
        n_dmd_off = 3 * n_dmd_pre_trigger
        n_dmd_on = int(np.floor((n_cam_all_pixels_exposure - n_dmd_off) / n_display_patterns_frame))

    # check if DMD on time is supported
    if n_dmd_on < n_dmd_pre_trigger:
        raise ValueError(f"number of DMD on frames < number of DMD pre-trigger frames, "
                         f"{n_dmd_on:d}<{n_dmd_pre_trigger:d}")

    if n_dmd_off < n_dmd_pre_trigger:
        raise ValueError(f"number of DMD off frames < number of DMD pre-trigger frames, "
                         f"{n_dmd_off:d}<{n_dmd_pre_trigger:d}")

    # ######################################
    # set total time and create digital array
    # ######################################
    n_active = n_stabilize + n_frame * npatterns
    n_total = np.max([n_active, n_interval])

    do = np.zeros((n_total, n_digital_ch), dtype=np.uint8)

    # initialize with values from preset ...
    for k in preset["digital"].keys():
        do[:, daq_do_map[k]] = preset["digital"][k]

    # ######################################
    # turn off laser during long intervals (optional)
    # ######################################
    if turn_laser_off_during_interval:
        laser_line = [l for l, v in preset["digital"].items() if v and match(".*_laser", l)][0]
        do[n_active:, daq_do_map[laser_line]] = 0

    # ######################################
    # advance analog
    # ######################################
    do[0, daq_do_map["analog_trigger"]] = 1

    # ######################################
    # shutter opens one delay time before imaging starts
    # ######################################
    do[:, daq_do_map["sim_shutter"]] = 0
    do[n_stabilize - n_sim_shutter_delay:, daq_do_map["sim_shutter"]] = 1

    # ######################################
    # trigger camera
    # ######################################
    n_cam_trig_width = n_cam_exposure

    if camera in ["cam1", "both"]:
        do[:n_active, daq_do_map["sim_cam_sync"]] = 0
        for ii in range(n_cam_trig_width):
            do[n_stabilize + ii:n_active:n_frame, daq_do_map["sim_cam_sync"]] = 1

    if camera in ["cam2", "both"]:
        # camera enable trigger (must be high for advance trigger)
        do[:, daq_do_map["odt_cam_enable"]] = 1  # photron/phantom camera

        do[:, daq_do_map["odt_cam_sync"]] = 0
        for ii in range(n_cam_trig_width):
            do[n_stabilize + ii:n_active:n_frame, daq_do_map["odt_cam_sync"]] = 1

    # ######################################
    # DMD enable trigger
    # ######################################
    do[:, dmd_enable] = 1

    # ######################################
    # DMD advance trigger
    # ######################################
    # number of steps for dmd pre-trigger
    do[:, dmd_advance] = 0

    for ii in range(n_trig_width):
        for jj in range(n_display_patterns_frame):
            # display SIM pattern
            # warmup offset time - pretrigger time + offset between sub-frame patterns + offset for trigger display
            on_start_index = n_stabilize + n_readout - n_dmd_pre_trigger
            # id_now = on_start_index + jj * n_pattern_frame + ii
            id_now = on_start_index + jj * n_dmd_on + ii
            do[id_now:n_active:n_frame, dmd_advance] = 1

            if use_dmd_as_shutter:
                # display OFF pattern. Only display between frames, i.e. not between sub-frame patterns
                # off_start_index = on_start_index + (n_cam_exposure - n_readout) + ii
                off_start_index = on_start_index + n_display_patterns_frame * n_dmd_on + ii
                do[off_start_index:n_active:n_frame, dmd_advance] = 1

    # ######################################
    # monitor lines
    # ######################################
    do[:, daq_do_map["signal_monitor"]] = do[:, dmd_advance]
    do[:, daq_do_map["camera_trigger_monitor"]] = np.logical_or(do[:, daq_do_map["sim_cam_sync"]],
                                                                do[:, daq_do_map["odt_cam_sync"]])

    # ######################################
    # analog channels
    # ######################################
    ao = np.zeros((1, n_analog_ch), dtype=float)
    for k in preset["analog"].keys():
        ao[:, daq_ao_map[k]] = preset["analog"][k]

    info += f"sim channel stabilize time = {n_stabilize * dt * 1e3:.3f}ms = {n_stabilize:d} clock cycles\n"
    info += f"sim exposure time = {n_cam_exposure * dt * 1e3:.3f}ms = {n_cam_exposure:d} clock cycles\n"
    info += f"sim one frame = {n_frame * dt * 1e3:.3f}ms = {n_frame:d} clock cycles\n"
    info += f"sim one channel= {n_active * dt * 1e3:.3f}ms = {n_active:d} clock cycles\n"

    return do, ao, info


def get_generic_sequence(daq_do_map: dict,
                         daq_ao_map: dict,
                         preset,
                         exposure_time: float,
                         npatterns: int,
                         dt: float = 105e-6,
                         dmd_delay: float = 105e-6,
                         interval: float = 0,
                         n_trig_width: int = 1,
                         min_frame_time: float = 0,
                         cam_readout_time: float = 0,
                         use_dmd_as_shutter: bool = True):
    # todo: ...
    pass
