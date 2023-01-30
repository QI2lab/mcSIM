"""
Create DAQ program for SIM/ODT experiment

Relies on DAQ line mapping scheme used in daq.py and daq_map.py
"""

import numpy as np

def get_sim_odt_sequence(daq_do_map: dict,
                         daq_ao_map: dict,
                         presets: dict,
                         channels: list[str],
                         odt_exposure_time: float,
                         sim_exposure_time: float,
                         npatterns: int,
                         dt: float = 105e-6,
                         interval: float = 0,
                         n_odt_per_sim: int = 1,
                         n_trig_width: int = 1,
                         dmd_delay: float = 105e-6,
                         odt_stabilize_t: float = 0.,
                         min_odt_frame_time: float = 8e-3,
                         sim_readout_time: float = 10e-3,
                         sim_stabilize_t: float = 200e-3,
                         shutter_delay_time: float = 50e-3,
                         z_voltages: list[float] = None,
                         use_dmd_as_odt_shutter: bool =False,
                         n_digital_ch: int = 16,
                         n_analog_ch: int = 4,
                         acquisition_mode: str = None):
    """
    Create DAQ program for SIM/ODT experiment

    # todo: maybe should pass around dictionaries with all settings for SIM/ODT separately?

    @param daq_do_map: e.g. from daq_map.py
    @param daq_ao_map: e.g. from daq_map.py
    @param channels:
    @param presets:
    @param odt_exposure_time: odt exposure time in s
    @param sim_exposure_time: sim exposure time in s
    @param npatterns: list which should match size of channels
    @param dt: daq time step
    @param interval: interval between images
    @param n_odt_per_sim: number of ODT images to take per each SIM image set
    @param n_trig_width: width of triggers
    @param dmd_delay:
    @param odt_stabilize_t:
    @param min_odt_frame_time:
    @param sim_readout_time:
    @param sim_stabilize_t:
    @param shutter_delay_time:
    @param z_voltages:
    @param bool use_dmd_as_odt_shutter:
    @param n_digital_ch:
    @param n_analog_ch:
    @param acquisition_mode: "sim", "both", or "average", applies only to SIM sequence
    @return:
    """

    if z_voltages is None:
        z_voltages = [0]

    if interval != 0 and (len(channels) != 1 or len(z_voltages) != 1):
        raise NotImplementedError("Interval is not implemented for more than one channel or for z-stacks")
    # todo: can easily implement interval if no z-stack, but otherwise difficult since don't have a way
    # to stop the daq after a certain number of repeats

    if acquisition_mode is None:
        acquisition_mode = ["sim"] * len(channels)

    if len(acquisition_mode) != len(channels):
        raise ValueError(f"len(acquisition_mode)` and len(channels) must be equal, but they were {len(acquisition_mode):d} and {len(channels):d}")

    if len(acquisition_mode) != len(npatterns):
        raise ValueError(f"len(acquisition_mode)` and len(npatterns) must be equal, but they were {len(acquisition_mode):d} and {len(npatterns):d}")

    # programs for each channel
    digital_pgms = []
    analog_pgms = []
    info = ""
    for ch, acq_mode, npat in zip(channels, acquisition_mode, npatterns):
        if ch == "odt":
            d, a, i = get_odt_sequence(daq_do_map,
                                       daq_ao_map,
                                       presets[ch],
                                       odt_exposure_time,
                                       npat,
                                       dt=dt,
                                       interval=interval,
                                       nrepeats=n_odt_per_sim,
                                       n_trig_width=n_trig_width,
                                       dmd_delay=dmd_delay,
                                       stabilize_t=odt_stabilize_t,
                                       min_frame_time=min_odt_frame_time,
                                       shutter_delay_time=shutter_delay_time,
                                       n_digital_ch=n_digital_ch,
                                       n_analog_ch=n_analog_ch,
                                       use_dmd_as_shutter=use_dmd_as_odt_shutter)

            info += i
            digital_pgms.append(d)
            analog_pgms.append(a)
        else:
            d, a, i = get_sim_sequence(daq_do_map,
                                       daq_ao_map,
                                       presets[ch],
                                       sim_exposure_time,
                                       npat,
                                       dt=dt,
                                       interval=interval,
                                       nrepeats=1,
                                       n_trig_width=n_trig_width,
                                       dmd_delay=dmd_delay,
                                       stabilize_t=sim_stabilize_t,
                                       min_frame_time=0,
                                       cam_readout_time=sim_readout_time,
                                       shutter_delay_time=shutter_delay_time,
                                       n_digital_ch=n_digital_ch,
                                       n_analog_ch=n_analog_ch,
                                       use_dmd_as_shutter=True,
                                       acquisition_mode=acq_mode)

            # if there is only one mode, keep SIM shutter open
            if len(channels) == 1:
                d[:, daq_do_map["sim_shutter"]] = 1

            info += i
            digital_pgms.append(d)
            analog_pgms.append(a)

    # for z-stack, digital pgm are just repeated. We don't need to do anything at all
    digital_pgm_full = np.vstack(digital_pgms)

    # analog pgms must be repeated with correct z voltages
    analog_pgms_one_z = np.vstack(analog_pgms)

    # check correct number of analog program steps and analog triggers
    if not analog_pgms_one_z.shape[0] == np.sum(digital_pgm_full[:, daq_do_map["analog_trigger"]]):
        raise AssertionError(f"size of analog program={analog_pgms_one_z.shape[0]:d}"
                             f" should equal number of analog triggers={np.sum(digital_pgm_full[:, daq_do_map['analog_trigger']]):d}")

    # todo: correct logic here ... e.g. if one channel not present or etc.
    # check number of patterns and number of camera triggers match
    # for odt
    # if np.sum(digital_pgm_full[:, daq_do_map["odt_cam"]]) // n_trig_width != (n_odt_patterns * n_odt_per_sim):
    #     raise ValueError("number of odt pics (%d) did not match DAQ program (%d)" %
    #                      (n_odt_patterns * n_odt_per_sim, np.sum(digital_pgm_full[:, daq_do_map["odt_cam"]]) // n_trig_width))

    # for SIM
    # if np.sum(digital_pgm_full[:, daq_do_map["sim_cam"]]) // n_trig_width != n_sim_patterns:
    #     raise ValueError(
    #         f"number of sim pics ({n_sim_patterns:d}) did not match"
    #         f" DAQ program {np.sum(digital_pgm_full[:, daq_do_map['sim_cam']]) // n_trig_width:d}")

    # check program and number of pictures match
    # if np.sum(digital_program[:, daq_do_map["odt_cam"]]) // n_trig_width != nodt_pics // ntimes // nz:
    #     raise ValueError("number of odt pics (%d) did not match DAQ program (%d)" %
    #                      (nodt_pics, np.sum(digital_program[:, daq_do_map["odt_cam"]]) // n_trig_width))

    # if np.sum(digital_program[:, daq_do_map["sim_cam"]]) // n_trig_width != nsim_pics // ntimes // nz:
    #     raise ValueError(f"number of sim pics ({nsim_pics:d}) did not match DAQ program "
    #                      f"{np.sum(digital_program[:, daq_do_map['sim_cam']]) // n_trig_width:d}")
    #
    # if np.sum(digital_program[:, daq_do_map["analog_trigger"]]) != len(channels):
    #     raise ValueError(f"number of analog triggers, {np.sum(digital_program[:, daq_do_map['analog_trigger']]):d},"
    #                      f" did not match {len(channels) * nz:d}")

    # get correct voltage for each step
    analog_pgms_per_z = []
    for v in z_voltages:
        pgm_temp = np.array(analog_pgms_one_z, copy=True)
        pgm_temp[:, daq_ao_map["z_stage"]] = v
        analog_pgms_per_z.append(pgm_temp)

    analog_pgm_full = np.vstack(analog_pgms_per_z)
    analog_pgm_full[:, daq_ao_map["z_stage_monitor"]] = analog_pgm_full[:, daq_ao_map["z_stage"]]

    # print information
    info += "channels are:" + " ".join(channels) + "\n"
    info += f"full digital program = {digital_pgm_full.shape[0] * dt * 1e3: 0.3f}ms = {digital_pgm_full.shape[0]:d} clock cycles\n"
    info += f"full analog program = {analog_pgm_full.shape[0]} steps"

    return digital_pgm_full, analog_pgm_full, info


def get_odt_sequence(daq_do_map: dict,
                     daq_ao_map: dict,
                     preset: dict,
                     exposure_time: float,
                     npatterns: int,
                     dt: float = 105e-6,
                     interval: float = 0,
                     nrepeats: int = 1,
                     n_trig_width: int = 1,
                     dmd_delay: float = 105e-6,
                     stabilize_t: float = 0.,
                     min_frame_time: float = 8e-3,
                     shutter_delay_time: float = 50e-3,
                     n_digital_ch: int = 16,
                     n_analog_ch: int = 4,
                     use_dmd_as_shutter: bool = False):
    """
    Get DAQ ODT sequence

    all times in seconds

    @param daq_do_map: dictionary with named lines as keys and line numbers as values. Must include lines ...
    @param daq_ao_map: dictionary with named lines as keys and line number as values.
    @param preset: preset used to set analog information. Dictionary with keys "analog" and "digital", where values
    are subentries. In this case, digital entries are ignored.
    @param exposure_time:
    @param npatterns: number of patterns
    @param dt: sampling time step
    @param interval:
    @param nrepeats: number of repeats. Typically used for time/z-stacks
    @param n_trig_width:
    @param dmd_delay: time to trigger DMD before triggering camera
    @param stabilize_t: time to wait for laser to stabilize at the start of sequence
    @param min_frame_time: minimum time before triggering the camera again
    @param shutter_delay_time: delay time to allow shutter to open
    @param n_digital_ch: number of digital lines
    @param n_analog_ch: number of analog lines
    @param use_dmd_as_shutter: whether to assume there are 2x as many DMD patterns with "off" pattern interleaved
    between on patterns
    @return: do_odt, ao_odt, info
    """

    info = ""

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
    do_odt[:, daq_do_map["dmd_enable"]] = 1
    # camera enable trigger (must be high for advance trigger)
    do_odt[:, daq_do_map["odt_cam_enable"]] = 1 # photron/phantom camera

    # set camera trigger, which starts after delay time for DMD to display pattern
    do_odt[n_odt_stabilize:nsteps_active:nsteps_frame, daq_do_map["odt_cam_sync"]] = 1
    # for debugging, make trigger pulse longer to see on scope
    for ii in range(n_trig_width):
        do_odt[n_odt_stabilize + ii:nsteps_active:nsteps_frame, daq_do_map["odt_cam_sync"]] = 1

    # DMD advance trigger
    do_odt[n_odt_stabilize - n_dmd_pre_trigger:nsteps_active-nsteps_frame:nsteps_frame, daq_do_map["dmd_advance"]] = 1

    # ending point is -nsteps_odt_frame to avoid having extra trigger at the end which is really the pretrigger for the next frame
    if use_dmd_as_shutter:
        # extra advance trigger to "turn off" DMD and end exposure
        do_odt[n_odt_stabilize - n_dmd_pre_trigger + nsteps_exposure:nsteps_active:nsteps_frame, daq_do_map["dmd_advance"]] = 1

    # monitor lines
    do_odt[:, daq_do_map["signal_monitor"]] = do_odt[:, daq_do_map["dmd_advance"]]
    do_odt[:, daq_do_map["camera_trigger_monitor"]] = do_odt[:, daq_do_map["odt_cam_sync"]]

    # set analog channels to match given preset
    ao_odt = np.zeros((1, n_analog_ch))
    for k in preset["analog"].keys():
        ao_odt[:, daq_ao_map[k]] = preset["analog"][k]

    # useful information to print
    info += f"odt stabilize time = {stabilize_t * 1e3:.3f}ms = {n_odt_stabilize:d} clock cycles\n"
    info += f"odt exposure time = {exposure_time * 1e3:.3f}ms = {nsteps_exposure:d} clock cycles\n"
    info += f"odt one frame = {nsteps_frame * dt * 1e3:.3f}ms = {nsteps_frame:d} clock cycles\n"
    info += f"odt one sequence of {nrepeats:d} volumes = {nsteps_active * dt * 1e3:.3f}ms = {nsteps_active:d} clock cycles\n"

    return do_odt, ao_odt, info


def get_sim_sequence(daq_do_map: dict,
                     daq_ao_map: dict,
                     preset: dict,
                     exposure_time: float,
                     npatterns: int,
                     dt: float = 105e-6,
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
                     acquisition_mode: str = "sim",
                     force_equal_subpatterns: bool = True):
    """
    Generate DAQ array for running a SIM experiment. Also supports taking a pseudo-widefield image by
    running through all SIM patterns during one camera exposure

    @param daq_do_map:
    @param daq_ao_map:
    @param preset:
    @param exposure_time:
    @param npatterns:
    @param dt:
    @param interval:
    @param nrepeats:
    @param n_trig_width:
    @param dmd_delay:
    @param stabilize_t:
    @param min_frame_time:
    @param cam_readout_time:
    @param shutter_delay_time:
    @param n_digital_ch:
    @param n_analog_ch:
    @param use_dmd_as_shutter:
    @param acquisition_mode: "sim", "both", or "average"
    @return do_channel, ao_channel, info:
    """

    info = ""

    if nrepeats != 1:
        raise NotImplementedError("only nrepeats=1 is implemented")

    if min_frame_time != 0:
        raise NotImplementedError("only min_frame_time=0 is implemented")

    # if using "average" mode, all patterns displayed during one frame time
    if acquisition_mode == "average":
        npatterns_frame = npatterns
        npatterns = 1
    else:
        npatterns_frame = 1

    # delay between frames
    # todo: implement ... should be as simple as in ODT function
    nsteps_interval = int(np.ceil(interval / dt))
    if interval != 0:
        raise NotImplementedError("only nsteps_interval=0 is implemented")

    # time for camera to roll open/closed
    n_readout = int(np.round(cam_readout_time / dt))

    # exposure time
    nsteps_exposure = int(np.ceil(exposure_time / dt))

    if force_equal_subpatterns:
        nsteps_exposure += nsteps_exposure % (npatterns_frame + 1)

    # sub-frame pattern time
    nsteps_pattern_frame = int(np.floor(nsteps_exposure / (npatterns_frame + 1)))

    # frame time
    nsteps_frame = nsteps_exposure + 2 * n_readout

    # time to stabilize (laser power/mirror)
    n_stabilize = int(np.ceil(stabilize_t / dt))

    n_sim_shutter_delay = int(np.ceil(shutter_delay_time / dt))
    if n_stabilize - n_sim_shutter_delay < 0:
        n_sim_shutter_delay = 0

    nsteps = n_stabilize + nsteps_frame * npatterns

    # ######################################
    # create digital array
    # ######################################
    do = np.zeros((nsteps, n_digital_ch), dtype=np.uint8)

    # initialize with values from preset ...
    for k in preset["digital"].keys():
        do[:, daq_do_map[k]] = preset["digital"][k]

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
    if acquisition_mode == "sim" or acquisition_mode == "average":
        do[:, daq_do_map["sim_cam_sync"]] = 0
        for ii in range(n_trig_width):
            do[n_stabilize + ii::nsteps_frame, daq_do_map["sim_cam_sync"]] = 1

    elif acquisition_mode == "both":
        do[:, daq_do_map["sim_cam_sync"]] = 0
        for ii in range(n_trig_width):
            do[n_stabilize + ii::nsteps_frame, daq_do_map["sim_cam_sync"]] = 1

        do[:, daq_do_map["odt_cam_sync"]] = 0
        for ii in range(n_trig_width):
            do[n_stabilize + ii::nsteps_frame, daq_do_map["odt_cam_sync"]] = 1

    else:
        raise ValueError(f"camera must be `sim` `both`, or 'average' but was `{acquisition_mode:s}`")


    # ######################################
    # DMD enable trigger
    # ######################################
    do[:, daq_do_map["dmd_enable"]] = 1

    # ######################################
    # DMD advance trigger
    # ######################################
    # number of steps for dmd pre-trigger
    do[:, daq_do_map["dmd_advance"]] = 0

    n_dmd_pre_trigger = int(np.round(dmd_delay / dt))
    for ii in range(n_trig_width):
        for jj in range(npatterns_frame):

            # display SIM pattern
            # do[n_stabilize + n_readout - n_dmd_pre_trigger + ii::nsteps_frame, daq_do_map["dmd_advance"]] = 1
            # warmup offset time - pretrigger time + offset between sub-frame patterns + offset for trigger display
            start_index = n_stabilize + n_readout - n_dmd_pre_trigger + jj * nsteps_pattern_frame + ii

            do[start_index::nsteps_frame, daq_do_map["dmd_advance"]] = 1

            if use_dmd_as_shutter:
                # display OFF pattern (only between frames)
                do[n_stabilize + n_readout - n_dmd_pre_trigger + (nsteps_exposure - n_readout) + ii::nsteps_frame, daq_do_map["dmd_advance"]] = 1


    # ######################################
    # monitor lines
    # ######################################
    do[:, daq_do_map["signal_monitor"]] = do[:, daq_do_map["dmd_advance"]]
    do[:, daq_do_map["camera_trigger_monitor"]] = do[:, daq_do_map["sim_cam_sync"]]

    # ######################################
    # analog channels
    # ######################################
    ao = np.zeros((1, n_analog_ch), dtype=float)
    for k in preset["analog"].keys():
        ao[:, daq_ao_map[k]] = preset["analog"][k]

    info += f"sim channel stabilize time = {n_stabilize * dt * 1e3:.3f}ms = {n_stabilize:d} clock cycles\n"
    info += f"sim exposure time = {nsteps_exposure * dt * 1e3:.3f}ms = {nsteps_exposure:d} clock cycles\n"
    info += f"sim one frame = {nsteps_frame * dt * 1e3:.3f}fms = {nsteps_frame:d} clock cycles\n"
    info += f"sim one channel= {nsteps * dt * 1e3:.3f}fms = {nsteps:d} clock cycles\n"

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
