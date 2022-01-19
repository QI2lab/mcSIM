import numpy as np

def build_odt_sim_sequence(daq_do_map, daq_ao_map, sim_channels):
    use_dmd_as_odt_shutter = True

    # timing variables
    dt = 105e-6 # s
    daq_sample_rate_hz = 1 / dt

    # dmd pre-trigger
    dmd_delay = 105e-6
    n_dmd_pre_trigger = int(np.round(dmd_delay / dt))

    # delay between frames
    delay_between_frames = 0 # s
    nsteps_delay = int(np.ceil(delay_between_frames / dt))
    if delay_between_frames != 0:
        raise NotImplementedError()

    # minimum frame time
    min_frame_time = 15e-3 # s
    nsteps_min_frame = int(np.ceil(min_frame_time / dt))

    # odt information
    odt_exposure_time = 2.8e-3 # s
    n_odt_patterns = 11

    # sim information
    sim_exposure_time = 100e-3 # s
    n_sim_patterns = 9
    n_sim_channels = 3

    # other information
    n_times = 1
    n_odt_per_sim = 5

    # trigger width
    t_trig_width = 10e-3
    n_trig_width = int(np.ceil(t_trig_width / dt))

    if nsteps_min_frame == 1 and use_dmd_as_odt_shutter:
        raise ValueError("nsteps_min_frame must be > 1 if use_dmd_as_odt_shutter=True")

    # #########################
    # create ODT program
    # #########################
    # calculate number of clock steps for different pieces ...
    nsteps_odt_exposure = int(np.ceil(odt_exposure_time / dt))
    nsteps_odt_frame = np.max([nsteps_odt_exposure, nsteps_min_frame])

    # time for laser power to stabilize
    odt_stabilize_t = 200e-3
    n_odt_stabilize = int(np.ceil(odt_stabilize_t / dt))

    shutter_odt_delay_t = 50e-3
    n_odt_shutter_delay = int(np.ceil(shutter_odt_delay_t / dt))
    if n_odt_stabilize - n_odt_shutter_delay < 0:
        n_odt_shutter_delay = 0


    nsteps_odt = nsteps_odt_frame * n_odt_patterns * n_odt_per_sim + n_odt_stabilize
    do_odt = np.zeros((nsteps_odt, 16), dtype=np.uint8)

    print("odt stabilize time = %0.2fms = %d clock cycles" % (odt_stabilize_t * 1e3, n_odt_stabilize))
    print("odt exposure time = %0.2fms = %d clock cycles" % (odt_exposure_time * 1e3, nsteps_odt_exposure))
    print("odt one frame = %0.2fms = %d clock cycles" % (nsteps_odt_frame * dt * 1e3, nsteps_odt_frame))
    print("odt one sequence of %d volumes = %0.2fms = %d clock cycles" % (n_odt_per_sim, nsteps_odt * dt * 1e3, nsteps_odt))

    # shutter always on
    do_odt[n_odt_stabilize - n_odt_shutter_delay:, daq_do_map["odt_shutter"]] = 1
    # laser always on
    do_odt[:, daq_do_map["odt_laser"]] = 1
    # DMD enable trigger always on
    do_odt[:, daq_do_map["dmd_enable"]] = 1

    # set camera trigger, which starts after delay time for DMD to display pattern
    do_odt[n_odt_stabilize::nsteps_odt_frame, daq_do_map["odt_cam"]] = 1
    # for debugging, make trigger pulse longer so i can see it on scope
    for ii in range(n_trig_width):
        do_odt[n_odt_stabilize + ii::nsteps_odt_frame, daq_do_map["odt_cam"]] = 1

    # DMD advance trigger
    do_odt[n_odt_stabilize - n_dmd_pre_trigger::nsteps_odt_frame, daq_do_map["dmd_advance"]] = 1
    if use_dmd_as_odt_shutter:
        # extra advance trigger to "turn off" DMD and end exposure
        do_odt[n_odt_stabilize - n_dmd_pre_trigger + nsteps_odt_exposure::nsteps_odt_frame, daq_do_map["dmd_advance"]] = 1

    # ao
    ao_odt = np.zeros((nsteps_odt, 3))

    # #########################
    # create SIM program
    # #########################
    nsteps_sim_exposure = int(np.ceil(sim_exposure_time / dt))
    nsteps_sim_frame = np.max([nsteps_sim_exposure, nsteps_min_frame])

    # time for camera to roll open/closed
    roll_open_t = 15e-3 # s
    n_roll_open = int(np.round(roll_open_t / dt))
    n_keep_on = nsteps_sim_exposure - 2 * n_roll_open

    # time for channel to stabilize (laser power/mirror)
    channel_stabilize_t = 200e-3
    n_sim_stabilize = int(np.ceil(channel_stabilize_t / dt))

    shutter_sim_delay_t = 50e-3
    n_sim_shutter_delay = int(np.ceil(shutter_sim_delay_t / dt))
    if n_sim_stabilize - n_sim_shutter_delay < 0:
        n_sim_shutter_delay = 0

    nsteps_sim_channel = n_sim_stabilize + nsteps_sim_frame * n_sim_patterns

    # create channel
    do_sim_channel = np.zeros((nsteps_sim_channel, 16), dtype=np.uint8)

    # shutter opens one delay time before imaging starts
    do_sim_channel[n_sim_stabilize - n_sim_shutter_delay:, daq_do_map["sim_shutter"]] = 1
    # DMD always enabled
    do_sim_channel[:, daq_do_map["dmd_enable"]] = 1
    # trigger camera
    for ii in range(n_trig_width):
        do_sim_channel[n_sim_stabilize + ii::nsteps_sim_frame, daq_do_map["sim_cam"]] = 1

    # only want DMD on when full camera is on
    for ii in range(n_trig_width):
        do_sim_channel[n_sim_stabilize + n_roll_open - n_dmd_pre_trigger + ii::nsteps_sim_frame, daq_do_map["dmd_advance"]] = 1 # display SIM pattern
        do_sim_channel[n_sim_stabilize + n_roll_open - n_dmd_pre_trigger + n_keep_on + ii::nsteps_sim_frame, daq_do_map["dmd_advance"]] = 1 # display OFF pattern

    ao_sim_channel = np.zeros((nsteps_sim_channel, 3))

    # assemble channels
    do_sim_blue = np.array(do_sim_channel, copy=True)
    do_sim_blue[:, daq_do_map["blue_laser"]] = 1

    ao_sim_blue = np.array(ao_sim_channel, copy=True)
    ao_sim_blue[:, daq_ao_map["vc_mirror_x"]] = 0
    ao_sim_blue[:, daq_ao_map["vc_mirror_y"]] = 0

    do_sim_red = np.array(do_sim_channel, copy=True)
    do_sim_red[:, daq_do_map["red_laser"]] = 1
    ao_sim_red = np.array(ao_sim_channel, copy=True)
    ao_sim_red[:, daq_ao_map["vc_mirror_x"]] = 1
    ao_sim_red[:, daq_ao_map["vc_mirror_y"]] = 1

    do_sim_green = np.array(do_sim_channel, copy=True)
    do_sim_green[:, daq_do_map["green_laser"]] = 1
    ao_sim_green = np.array(ao_sim_channel, copy=True)
    ao_sim_green[:, daq_ao_map["vc_mirror_x"]] = 2
    ao_sim_green[:, daq_ao_map["vc_mirror_y"]] = 2

    do_sim = np.vstack((do_sim_blue, do_sim_red, do_sim_green))
    ao_sim = np.vstack((ao_sim_blue, ao_sim_red, ao_sim_green))
    nsteps_sim = do_sim.shape[0]


    print("sim channel stabilize time = %0.2fms = %d clock cycles" % (n_sim_stabilize * dt * 1e3, n_sim_stabilize))
    print("sim exposure time = %0.2fms = %d clock cycles" % (nsteps_sim_exposure * dt * 1e3, nsteps_sim_exposure))
    print("sim one frame = %0.2fms = %d clock cycles" % (nsteps_sim_frame * dt * 1e3, nsteps_sim_frame))
    print("sim one channel= %0.2fms = %d clock cycles" % (nsteps_sim_channel * dt * 1e3, nsteps_sim_channel))
    print("sim %d channels = %0.2fms = %d clock cycles" % (3, nsteps_sim * dt * 1e3, nsteps_sim))


    # #########################
    # complete program
    # #########################
    do_sim_odt = np.vstack((do_odt, do_sim))
    ao_sim_odt = np.vstack((ao_odt, ao_sim))
    assert do_sim_odt.shape[0] == ao_sim_odt.shape[0]

    nsteps_pgm = do_sim_odt.shape[0]

    print("full program = %0.2fms = %d clock cycles" % (nsteps_pgm * dt * 1e3, nsteps_pgm))

    return do_sim_odt, ao_sim_odt
