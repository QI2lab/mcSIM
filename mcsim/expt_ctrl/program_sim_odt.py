"""
Create DAQ program for SIM/ODT experiment
"""

import numpy as np
from mcsim.expt_ctrl.daq_map import presets

def build_odt_sim_sequence(daq_do_map, daq_ao_map, channels, odt_exposure_time, sim_exposure_time,
                           n_odt_patterns, n_sim_patterns, dt=105e-6, interval=0,
                           n_odt_per_sim=1, n_trig_width=1, dmd_delay=105e-6,
                           odt_stabilize_t=0., min_odt_frame_time=8e-3,
                           sim_readout_time=10e-3, shutter_delay_time=50e-3,
                           use_dmd_as_odt_shutter=False):
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

    # dmd pre-trigger
    n_dmd_pre_trigger = int(np.round(dmd_delay / dt))

    # delay between frames
    nsteps_delay = int(np.ceil(interval / dt))
    if interval != 0:
        raise NotImplementedError("non-zero interval between adjacent frames not yet implemented")

    # minimum frame time
    nsteps_min_frame = int(np.ceil(min_odt_frame_time / dt))

    if nsteps_min_frame == 1 and use_dmd_as_odt_shutter:
        raise ValueError("nsteps_min_frame must be > 1 if use_dmd_as_odt_shutter=True")

    # #########################
    # create ODT program
    # #########################
    # calculate number of clock steps for different pieces ...
    nsteps_odt_exposure = int(np.ceil(odt_exposure_time / dt))
    nsteps_odt_frame = np.max([nsteps_odt_exposure, nsteps_min_frame])

    # time for laser power to stabilize
    n_odt_stabilize = int(np.ceil(odt_stabilize_t / dt))
    if n_odt_stabilize < n_dmd_pre_trigger:
        n_odt_stabilize = n_dmd_pre_trigger
        print("n_odt_stabilize was less than n_dmd_pre_trigger. Increased it to %d steps" % n_odt_stabilize)

    n_odt_shutter_delay = int(np.ceil(shutter_delay_time / dt))
    if n_odt_stabilize - n_odt_shutter_delay < 0:
        n_odt_shutter_delay = 0

    # if n_odt_stabilize < n_dmd_pre_trigger:
    #     raise ValueError("number of stabilization steps must be greater than n_dmd_pre_trigger")

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
    # master trigger
    do_odt[1:, daq_do_map["odt_cam_master_trig"]] = 1

    # set camera trigger, which starts after delay time for DMD to display pattern
    do_odt[n_odt_stabilize::nsteps_odt_frame, daq_do_map["odt_cam"]] = 1
    # for debugging, make trigger pulse longer so i can see it on scope
    for ii in range(n_trig_width):
        do_odt[n_odt_stabilize + ii::nsteps_odt_frame, daq_do_map["odt_cam"]] = 1

    # DMD advance trigger
    do_odt[n_odt_stabilize - n_dmd_pre_trigger:-nsteps_odt_frame:nsteps_odt_frame, daq_do_map["dmd_advance"]] = 1
    # ending point is -nsteps_odt_frame to avoid having extra trigger at the end which is really the "pretrigger"
    # for the next frame
    if use_dmd_as_odt_shutter:
        # extra advance trigger to "turn off" DMD and end exposure
        do_odt[n_odt_stabilize - n_dmd_pre_trigger + nsteps_odt_exposure::nsteps_odt_frame, daq_do_map["dmd_advance"]] = 1

    # ao
    ao_odt = np.zeros((nsteps_odt, 3))

    # #########################
    # create SIM program
    # #########################
    # time for camera to roll open/closed
    n_roll_open = int(np.round(sim_readout_time / dt))

    # exposure time
    nsteps_sim_exposure = int(np.ceil(sim_exposure_time / dt))
    nsteps_sim_frame = nsteps_sim_exposure + 2 * n_roll_open

    # time for channel to stabilize (laser power/mirror)
    channel_stabilize_t = 200e-3
    n_sim_stabilize = int(np.ceil(channel_stabilize_t / dt))

    n_sim_shutter_delay = int(np.ceil(shutter_delay_time / dt))
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
        do_sim_channel[n_sim_stabilize + n_roll_open - n_dmd_pre_trigger + (nsteps_sim_exposure - n_roll_open) + ii::nsteps_sim_frame, daq_do_map["dmd_advance"]] = 1 # display OFF pattern

    ao_sim_channel = np.zeros((nsteps_sim_channel, 3))

    # assemble channels
    do_sim_blue = np.array(do_sim_channel, copy=True)
    do_sim_blue[:, daq_do_map["blue_laser"]] = 1

    ao_sim_blue = np.array(ao_sim_channel, copy=True)
    ao_sim_blue[:, daq_ao_map["vc_mirror_x"]] = presets["blue"]["analog"]["vc_mirror_x"]
    ao_sim_blue[:, daq_ao_map["vc_mirror_y"]] = presets["blue"]["analog"]["vc_mirror_y"]

    do_sim_red = np.array(do_sim_channel, copy=True)
    do_sim_red[:, daq_do_map["red_laser"]] = 1
    ao_sim_red = np.array(ao_sim_channel, copy=True)
    ao_sim_red[:, daq_ao_map["vc_mirror_x"]] = presets["red"]["analog"]["vc_mirror_x"]
    ao_sim_red[:, daq_ao_map["vc_mirror_y"]] = presets["red"]["analog"]["vc_mirror_y"]

    do_sim_green = np.array(do_sim_channel, copy=True)
    do_sim_green[:, daq_do_map["green_laser"]] = 1
    ao_sim_green = np.array(ao_sim_channel, copy=True)
    ao_sim_green[:, daq_ao_map["vc_mirror_x"]] = presets["green"]["analog"]["vc_mirror_x"]
    ao_sim_green[:, daq_ao_map["vc_mirror_y"]] = presets["green"]["analog"]["vc_mirror_y"]

    sim_daq_data_ch = {"blue": {"do": do_sim_blue, "ao": ao_sim_blue},
                       "red": {"do": do_sim_red, "ao": ao_sim_red},
                       "green": {"do": do_sim_green, "ao": ao_sim_green}}

    print("sim channel stabilize time = %0.2fms = %d clock cycles" % (n_sim_stabilize * dt * 1e3, n_sim_stabilize))
    print("sim exposure time = %0.2fms = %d clock cycles" % (nsteps_sim_exposure * dt * 1e3, nsteps_sim_exposure))
    print("sim one frame = %0.2fms = %d clock cycles" % (nsteps_sim_frame * dt * 1e3, nsteps_sim_frame))
    print("sim one channel= %0.2fms = %d clock cycles" % (nsteps_sim_channel * dt * 1e3, nsteps_sim_channel))


    # #########################
    # complete program
    # #########################

    # combine channels in correct order
    do_list = []
    ao_list = []
    for ch in channels:
        if ch == "odt":
            do_list.append(do_odt)
            ao_list.append(ao_odt)
        else:
            do_list.append(sim_daq_data_ch[ch]["do"])
            ao_list.append(sim_daq_data_ch[ch]["ao"])


    # build complete program
    do_sim_odt = np.vstack(do_list)
    ao_sim_odt = np.vstack(ao_list)

    assert do_sim_odt.shape[0] == ao_sim_odt.shape[0]

    nsteps_pgm = do_sim_odt.shape[0]

    print("channels are:", end="")
    print(" ".join(channels))

    print("full program = %0.2fms = %d clock cycles" % (nsteps_pgm * dt * 1e3, nsteps_pgm))

    return do_sim_odt, ao_sim_odt, dt
