"""
Create DAQ program for SIM/ODT experiment

Relies on DAQ line mapping scheme used in daq.py and daq_map.py
"""

import numpy as np

def get_sim_odt_sequence(daq_do_map, daq_ao_map, presets, channels, odt_exposure_time, sim_exposure_time,
                         n_odt_patterns, n_sim_patterns, dt=105e-6, interval=0,
                         n_odt_per_sim=1, n_trig_width=1, dmd_delay=105e-6,
                         odt_stabilize_t=0., min_odt_frame_time=8e-3,
                         sim_readout_time=10e-3, sim_stabilize_t=200e-3, shutter_delay_time=50e-3,
                         z_voltages=None,
                         use_dmd_as_odt_shutter=False, n_digital_ch=16, n_analog_ch=4):
    """
    Create DAQ program for SIM/ODT experiment

    @param dict daq_do_map: e.g. from daq_map.py
    @param dict daq_ao_map: e.g. from daq_map.py
    @param list[str] channels:
    @param float odt_exposure_time: odt exposure time in s
    @param float sim_exposure_time: sim exposure time in s
    @param int n_odt_patterns:
    @param int n_sim_patterns:
    @param float dt: daq time step
    @param float interval: interval between images
    @param int n_odt_per_sim: number of ODT images to take per each SIM image set
    @param int n_trig_width: width of triggers
    @param float dmd_delay:
    @param float odt_stabilize_t:
    @param bool use_dmd_as_odt_shutter:
    @return:
    """

    if z_voltages is None:
        z_voltages = [0]

    # programs for each channel
    digital_pgms = []
    analog_pgms = []
    for ch in channels:
        if ch == "odt":
            d, a = get_odt_sequence(daq_do_map, daq_ao_map, presets[ch], odt_exposure_time, n_odt_patterns,
                                    dt=dt, interval=0, nrepeats=n_odt_per_sim, n_trig_width=n_trig_width,
                                    dmd_delay=dmd_delay, stabilize_t=odt_stabilize_t, min_frame_time=min_odt_frame_time,
                                    shutter_delay_time=shutter_delay_time, n_digital_ch=n_digital_ch,
                                    n_analog_ch=n_analog_ch, use_dmd_as_shutter=use_dmd_as_odt_shutter)
        else:
            d, a = get_sim_sequence(daq_do_map, daq_ao_map, presets[ch], sim_exposure_time, n_sim_patterns,
                                    dt=dt, interval=0, nrepeats=1, n_trig_width=n_trig_width, dmd_delay=dmd_delay,
                                    stabilize_t=sim_stabilize_t, min_frame_time=0, cam_readout_time=sim_readout_time,
                                    shutter_delay_time=shutter_delay_time, n_digital_ch=n_digital_ch, n_analog_ch=n_analog_ch,
                                    use_dmd_as_shutter=True)

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

    # get correct voltage for each step
    analog_pgms_per_z = []
    for v in z_voltages:
        pgm_temp = np.array(analog_pgms_one_z, copy=True)
        pgm_temp[:, daq_ao_map["z_stage"]] = v
        analog_pgms_per_z.append(pgm_temp)

    analog_pgm_full = np.vstack(analog_pgms_per_z)
    analog_pgm_full[:, daq_ao_map["z_stage_monitor"]] = analog_pgm_full[:, daq_ao_map["z_stage"]]

    # print information
    print("channels are:", end="")
    print(" ".join(channels))

    print(f"full digital program = {digital_pgm_full.shape[0] * dt * 1e3: 0.3f}ms = {digital_pgm_full.shape[0]:d} clock cycles")
    print(f"full analog program = {analog_pgm_full.shape[0]} steps")

    return digital_pgm_full, analog_pgm_full


def get_odt_sequence(daq_do_map, daq_ao_map, preset,
                     exposure_time, npatterns, dt=105e-6, interval=0,
                     nrepeats=1, n_trig_width=1, dmd_delay=105e-6,
                     stabilize_t=0., min_frame_time=8e-3,
                     shutter_delay_time=50e-3, n_digital_ch=16, n_analog_ch=4,
                     use_dmd_as_shutter=False):
    """
    Get DAQ ODT sequence

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
    @param shutter_delay_time:
    @param n_digital_ch:
    @param n_analog_ch:
    @param use_dmd_as_shutter:
    @return:
    """

    # number of steps for dmd pre-trigger
    n_dmd_pre_trigger = int(np.round(dmd_delay / dt))

    # delay between frames
    nsteps_delay = int(np.ceil(interval / dt))
    if interval != 0:
        raise NotImplementedError("non-zero interval between adjacent frames not yet implemented")

    # minimum frame time
    nsteps_min_frame = int(np.ceil(min_frame_time / dt))

    if nsteps_min_frame == 1 and use_dmd_as_shutter:
        raise ValueError("nsteps_min_frame must be > 1 if use_dmd_as_odt_shutter=True")

    # #########################
    # create ODT program
    # #########################
    # calculate number of clock steps for different pieces ...
    nsteps_exposure = int(np.ceil(exposure_time / dt))
    nsteps_frame = np.max([nsteps_exposure, nsteps_min_frame])

    # time for laser power to stabilize
    n_odt_stabilize = int(np.ceil(stabilize_t / dt))
    if n_odt_stabilize < n_dmd_pre_trigger:
        n_odt_stabilize = n_dmd_pre_trigger
        print("n_odt_stabilize was less than n_dmd_pre_trigger. Increased it to %d steps" % n_odt_stabilize)

    # shutter delay
    n_odt_shutter_delay = int(np.ceil(shutter_delay_time / dt))
    if n_odt_stabilize - n_odt_shutter_delay < 0:
        n_odt_shutter_delay = 0

    # if n_odt_stabilize < n_dmd_pre_trigger:
    #     raise ValueError("number of stabilization steps must be greater than n_dmd_pre_trigger")

    nsteps_odt = nsteps_frame * npatterns * nrepeats + n_odt_stabilize
    do_odt = np.zeros((nsteps_odt, n_digital_ch), dtype=np.uint8)

    print("odt stabilize time = %0.3fms = %d clock cycles" % (stabilize_t * 1e3, n_odt_stabilize))
    print("odt exposure time = %0.3fms = %d clock cycles" % (exposure_time * 1e3, nsteps_exposure))
    print("odt one frame = %0.3fms = %d clock cycles" % (nsteps_frame * dt * 1e3, nsteps_frame))
    print("odt one sequence of %d volumes = %0.3fms = %d clock cycles" % (nrepeats, nsteps_odt * dt * 1e3, nsteps_odt))

    # trigger analog lines to start
    do_odt[0, daq_do_map["analog_trigger"]] = 1
    # shutter on one delay time before imaging
    do_odt[n_odt_stabilize - n_odt_shutter_delay:, daq_do_map["odt_shutter"]] = 1
    # laser always on
    do_odt[:, daq_do_map["odt_laser"]] = 1
    # DMD enable trigger always on
    do_odt[:, daq_do_map["dmd_enable"]] = 1
    # master trigger
    do_odt[:, daq_do_map["odt_cam_master_trig"]] = 1 # photron/phantom camera


    # set camera trigger, which starts after delay time for DMD to display pattern
    do_odt[n_odt_stabilize::nsteps_frame, daq_do_map["odt_cam"]] = 1
    # for debugging, make trigger pulse longer to see on scope
    for ii in range(n_trig_width):
        do_odt[n_odt_stabilize + ii::nsteps_frame, daq_do_map["odt_cam"]] = 1

    # DMD advance trigger
    do_odt[n_odt_stabilize - n_dmd_pre_trigger:-nsteps_frame:nsteps_frame, daq_do_map["dmd_advance"]] = 1

    # ending point is -nsteps_odt_frame to avoid having extra trigger at the end which is really the pretrigger for the next frame
    if use_dmd_as_shutter:
        # extra advance trigger to "turn off" DMD and end exposure
        do_odt[n_odt_stabilize - n_dmd_pre_trigger + nsteps_exposure::nsteps_frame, daq_do_map["dmd_advance"]] = 1

    # invert camera trigger line
    #do_odt[:, daq_do_map["odt_cam"]] = 1 - do_odt[:, daq_do_map["odt_cam"]]

    # monitor lines
    do_odt[:, daq_do_map["signal_monitor"]] = do_odt[:, daq_do_map["dmd_advance"]]
    do_odt[:, daq_do_map["camera_trigger_monitor"]] = do_odt[:, daq_do_map["odt_cam"]]
    # do_odt[:, daq_do_map["camera_trigger_monitor"]] = do_odt[:, daq_do_map["odt_cam_master_trig"]]

    # set analog channels
    ao_odt = np.zeros((1, n_analog_ch))
    for k in preset["analog"].keys():
        ao_odt[:, daq_ao_map[k]] = preset["analog"][k]

    return do_odt, ao_odt


def get_sim_sequence(daq_do_map, daq_ao_map, preset,
                     exposure_time, npatterns, dt=105e-6, interval=0,
                     nrepeats=1, n_trig_width=1, dmd_delay=105e-6,
                     stabilize_t=200e-3, min_frame_time=0, cam_readout_time=10e-3,
                     shutter_delay_time=50e-3, n_digital_ch=16, n_analog_ch=4,
                     use_dmd_as_shutter=True):

    if nrepeats != 1:
        raise NotImplementedError()

    if min_frame_time != 0:
        raise NotImplementedError()

    # delay between frames
    nsteps_delay = int(np.ceil(interval / dt))
    if interval != 0:
        raise NotImplementedError("non-zero interval between adjacent frames not yet implemented")

    # time for camera to roll open/closed
    n_roll_open = int(np.round(cam_readout_time / dt))

    # exposure time
    nsteps_sim_exposure = int(np.ceil(exposure_time / dt))
    nsteps_sim_frame = nsteps_sim_exposure + 2 * n_roll_open

    # time for channel to stabilize (laser power/mirror)
    n_sim_stabilize = int(np.ceil(stabilize_t / dt))

    n_sim_shutter_delay = int(np.ceil(shutter_delay_time / dt))
    if n_sim_stabilize - n_sim_shutter_delay < 0:
        n_sim_shutter_delay = 0

    nsteps_sim_channel = n_sim_stabilize + nsteps_sim_frame * npatterns

    # ######################################
    # create channel
    # ######################################
    do_sim_channel = np.zeros((nsteps_sim_channel, n_digital_ch), dtype=np.uint8)

    # initialize with values from preset ...
    for k in preset["digital"].keys():
        do_sim_channel[:, daq_do_map[k]] = preset["digital"][k]

    # ######################################
    # advance analog
    # ######################################
    do_sim_channel[0, daq_do_map["analog_trigger"]] = 1

    # ######################################
    # shutter opens one delay time before imaging starts
    # ######################################
    do_sim_channel[:, daq_do_map["sim_shutter"]] = 0
    do_sim_channel[n_sim_stabilize - n_sim_shutter_delay:, daq_do_map["sim_shutter"]] = 1

    # ######################################
    # trigger camera
    # ######################################
    do_sim_channel[:, daq_do_map["sim_cam"]] = 0
    for ii in range(n_trig_width):
        do_sim_channel[n_sim_stabilize + ii::nsteps_sim_frame, daq_do_map["sim_cam"]] = 1

    # ######################################
    # DMD always enabled
    # ######################################
    do_sim_channel[:, daq_do_map["dmd_enable"]] = 1

    # ######################################
    # DMD advance triggering
    # ######################################
    # number of steps for dmd pre-trigger
    do_sim_channel[:, daq_do_map["dmd_advance"]] = 0

    n_dmd_pre_trigger = int(np.round(dmd_delay / dt))
    for ii in range(n_trig_width):
        do_sim_channel[n_sim_stabilize + n_roll_open - n_dmd_pre_trigger + ii::nsteps_sim_frame, daq_do_map["dmd_advance"]] = 1 # display SIM pattern

        if use_dmd_as_shutter:
            do_sim_channel[n_sim_stabilize + n_roll_open - n_dmd_pre_trigger + (nsteps_sim_exposure - n_roll_open) + ii::nsteps_sim_frame, daq_do_map["dmd_advance"]] = 1 # display OFF pattern

    # ######################################
    # monitor lines
    # ######################################
    do_sim_channel[:, daq_do_map["signal_monitor"]] = do_sim_channel[:, daq_do_map["dmd_advance"]]
    do_sim_channel[:, daq_do_map["camera_trigger_monitor"]] = do_sim_channel[:, daq_do_map["sim_cam"]]

    # ######################################
    # analog channels
    # ######################################
    ao_sim_channel = np.zeros((1, n_analog_ch), dtype=float)
    for k in preset["analog"].keys():
        ao_sim_channel[:, daq_ao_map[k]] = preset["analog"][k]

    print("sim channel stabilize time = %0.3fms = %d clock cycles" % (n_sim_stabilize * dt * 1e3, n_sim_stabilize))
    print("sim exposure time = %0.3fms = %d clock cycles" % (nsteps_sim_exposure * dt * 1e3, nsteps_sim_exposure))
    print("sim one frame = %0.3fms = %d clock cycles" % (nsteps_sim_frame * dt * 1e3, nsteps_sim_frame))
    print("sim one channel= %0.3fms = %d clock cycles" % (nsteps_sim_channel * dt * 1e3, nsteps_sim_channel))

    return do_sim_channel, ao_sim_channel


