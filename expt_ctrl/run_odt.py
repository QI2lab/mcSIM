"""
Script intended to be called from MM beanshell

In beanshell, I remove the NI devices, then run this script, then afterwards re-add the NI devices.
This is a workaround to use MM for cruising around the sample but PyDaqMx for controlling the DAQ
"""
import time
import os
import datetime
import ctypes as ct
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pycromanager as pm
import PyDAQmx as daq
import dlp6500

tstart = time.perf_counter()

use_dmd_as_shutter = True
# #########################
# turn laser and shutter on and allow laser power to stabilize
# ensure both DMD triggers are low before programming DMD
# #########################
# use this if device won't restart
# daq.DAQmxResetDevice("Dev1")
t_stabilize = 15 # ensure stabilization is longer than this...

taskDO = daq.Task()
taskDO.CreateDOChan("/Dev1/port0/line0:7", "", daq.DAQmx_Val_ChanForAllLines)
taskDO.WriteDigitalLines(1, 1, 10.0, daq.DAQmx_Val_GroupByChannel,
                         np.array([0, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8), None, None)
taskDO.StopTask()
taskDO.ClearTask()

print("initialized DAQ lines, elapsed time = %0.2fs" % (time.perf_counter() - tstart))

# #########################
# load dmd
# #########################
dmd = dlp6500.dlp6500win(debug=False)

dmd.start_stop_sequence('stop')

# check DMD trigger state
delay1_us, mode_trig1 = dmd.get_trigger_in1()
print('trigger1 delay=%dus' % delay1_us)
print('trigger1 mode=%d' % mode_trig1)

# dmd.set_trigger_in2('rising')
mode_trig2 = dmd.get_trigger_in2()
print("trigger2 mode=%d" % mode_trig2)

# dmd.set_pattern_sequence([0, 0], [0, 1], 105, 0, triggered=True,
#                          clear_pattern_after_trigger=False, bit_depth=1, num_repeats=0, mode='pre-stored')

dmd_size = [1080, 1920]
ny, nx = dmd_size
xx, yy = np.meshgrid(range(nx), range(ny))
cref = np.array([ny // 2, nx // 2])
xx, yy = np.meshgrid(range(nx), range(ny))

ang = -45 * np.pi/180
frq = np.array([np.sin(ang), np.cos(ang)]) * 1/4 * np.sqrt(2)

rad = 5
phase = 0
pattern_base = np.round(np.cos(2 * np.pi * (xx * frq[0] + yy * frq[1]) + phase), 12)
pattern_base[pattern_base <= 0] = 0
pattern_base[pattern_base > 0] = 1
pattern_base = 1 - pattern_base
print("pattern radius = %0.2f mirrors" % rad)
print("pattern frequency (fx, fy) = (%0.3f, %0.3f) 1/ mirrors" % tuple(frq))

# pupil info
na_mitutoyo = 0.55
dm = 7.56 # DMD mirror size
fl_mitutoyo = 4e3 # focal length of mitutoya objective
fl_olympus = 1.8e3
# magnification between DMD and Mitutoyo BFP
mag_dmd2bfp = 100 / 200 * 300 / 400 * fl_mitutoyo / fl_olympus

pupil_rad_mirrors = fl_mitutoyo * na_mitutoyo / mag_dmd2bfp / dm

# center positions
# n_phis = 10
# fractions = [0.25, 0.5, 0.75, 0.9]
n_phis = 5
fractions = [0.5, 0.9]
# n_phis = 10
# fractions = [0.9]
phis = np.arange(n_phis) * 2*np.pi / n_phis
n_thetas = len(fractions)

xoffs = np.zeros((n_phis, n_thetas))
yoffs = np.zeros((n_phis, n_thetas))
for ii in range(n_phis):
    for jj in range(n_thetas):
        xoffs[ii, jj] = np.cos(phis[ii]) * pupil_rad_mirrors * fractions[jj]
        yoffs[ii, jj] = np.sin(phis[ii]) * pupil_rad_mirrors * fractions[jj]

xoffs = np.concatenate((np.array([0]), xoffs.ravel()))
yoffs = np.concatenate((np.array([0]), yoffs.ravel()))


npatterns = xoffs.size

pattern_info = {"frquency": frq, "radius": rad, "cref": cref, "phase": phase, "dmd_size": dmd_size,
                "xoffsets": xoffs, "yoffsets": yoffs, "npatterns": npatterns}

print("generating patterns...")
if use_dmd_as_shutter:
    print("loading extra patterns to use DMD as shutter")

    patterns = np.ones((2*npatterns, ny, nx), dtype=np.uint8)
    for ii in range(npatterns):
        patterns[2*ii] = np.copy(pattern_base)
        patterns[2*ii, np.sqrt((xx - cref[1] - xoffs.ravel()[ii]) ** 2 +
                             (yy - cref[0] - yoffs.ravel()[ii]) ** 2) > rad] = 1
        patterns[2*ii + 1] = 1
else:
    patterns = np.ones((npatterns, ny, nx), dtype=np.uint8)
    for ii in range(npatterns):
        patterns[ii] = np.copy(pattern_base)
        patterns[ii, np.sqrt((xx - cref[1] - xoffs.ravel()[ii])**2 +
                             (yy - cref[0] - yoffs.ravel()[ii])**2) > rad] = 1
print("finished generating patterns, elapsed time %0.2fs" % (time.perf_counter() - tstart))


img_inds, bit_inds = dmd.upload_pattern_sequence(patterns, 105, 0, triggered=True, clear_pattern_after_trigger=False)
# dmd.set_pattern_sequence(img_inds, bit_inds, 105, 0, triggered=True, clear_pattern_after_trigger=True)
print("loaded %d patterns, elapsed time %0.2fs" % (npatterns, time.perf_counter() - tstart))

# #########################
# turn DMD enable trigger on
# #########################
# use this if device won't restart
# daq.DAQmxResetDevice("Dev1")

taskDO = daq.Task()
taskDO.CreateDOChan("/Dev1/port0/line0:7", "", daq.DAQmx_Val_ChanForAllLines)
taskDO.WriteDigitalLines(1, 1, 10.0, daq.DAQmx_Val_GroupByChannel,
                         np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=np.uint8), None, None)
taskDO.StopTask()
taskDO.ClearTask()

print("initialized DAQ lines, elapsed time = %0.2fs" % (time.perf_counter() - tstart))

# #########################
# check lasers have had time to stabilize, otherwise wait longer
# #########################
t_waited = time.perf_counter() - tstart
if t_waited < t_stabilize:
    time.sleep(t_stabilize - t_waited)

# #########################
# pycromanager
# #########################
with pm.Bridge() as bridge:
    mmc = bridge.get_core()
    mm = bridge.get_studio()

    # set circular buffer size
    mmc.set_circular_buffer_memory_footprint(3000)

    # pull save dir from MDA
    acq_settings = mm.get_acquisition_manager().get_acquisition_settings()
    save_dir = os.path.join(acq_settings.root(), acq_settings.prefix())
    save_dir = mm.data().get_unique_save_directory(save_dir)

    # get number of time steps from MDA
    ntimes = acq_settings.num_frames()
    delay_time = acq_settings.interval_ms()

    # #########################
    # create program for digital output
    # #########################
    dmd_delay = 105e-6 # s
    # dmd_delay = 0
    dt = dmd_delay
    # dt = 21e-6
    # dt = 10e-6
    daq_sample_rate_hz = 1 / dt

    exposure_time = 2.8e-3 # s
    min_frame_time = 15e-3  # accounting for readout time
    # exposure_time = 0.100e-3  # s
    # min_frame_time = 0.100e-3  # accounting for readout time
    delay_between_frames = delay_time / 1e3

    # calculate number of clock steps for different pieces ...
    nsteps_exposure = int(np.ceil(exposure_time / dt))
    nsteps_min_frame = int(np.ceil(min_frame_time / dt))
    nsteps_delay = int(np.ceil(delay_between_frames / dt))

    # want DMD to have time to pre-trigger so pattern is displayed when camera comes on
    n_dmd_pre_trigger = int(np.round(dmd_delay / dt))
    nsteps_frame = np.max([nsteps_exposure + n_dmd_pre_trigger, nsteps_min_frame])
    nsteps_sequence = np.max([nsteps_frame * npatterns, nsteps_delay])


    print("exposure time = %0.2fms = %d clock cycles" % (exposure_time * 1e3, nsteps_exposure))
    print("one frame = %0.2fms = %d clock cycles" % (nsteps_frame * dt * 1e3, nsteps_frame))
    print("one time point = %0.2fms = %d clock cycles" % (nsteps_sequence * dt * 1e3, nsteps_sequence))
    print("nframes = %d" % ntimes)

    # generate DAQ array for digital lines
    samples_per_ch = nsteps_sequence
    data_do = np.zeros((samples_per_ch, 8), dtype=np.uint8)

    # shutter always on
    data_do[:, 1] = 1
    # laser always on
    data_do[:, 2] = 1
    # DMD enable trigger always on
    data_do[:, 4] = 1

    # DMD advance trigger
    data_do[:(npatterns * nsteps_frame):nsteps_frame, 3] = 1
    if use_dmd_as_shutter:
        # extra advance trigger to "turn off" DMD and end exposure
        data_do[nsteps_exposure - n_dmd_pre_trigger:(npatterns * nsteps_frame):nsteps_frame, 3] = 1

    # set camera trigger, which starts after delay time for DMD to display pattern
    data_do[n_dmd_pre_trigger:(npatterns * nsteps_frame):nsteps_frame, 0] = 1
    # for debugging, make trigger pulse longer so i can see it on scope
    for ii in range(26):
        data_do[n_dmd_pre_trigger + ii:(npatterns * nsteps_frame):nsteps_frame, 0] = 1

    if False:
        # plot digital line data for first frame
        figh = plt.figure(figsize=(16, 8))
        ax = plt.subplot(1, 1, 1)
        # nstop = 2 * nsteps_frame
        nstop = nsteps_sequence

        ax.set_title("Digital line data (first frame)")
        ax.plot(data_do[:nstop, 0], label="camera trigger")
        ax.plot(data_do[:nstop, 1], label="shutter")
        ax.plot(data_do[:nstop, 2], label="laser")
        ax.plot(data_do[:nstop, 3], label="adv trigger")
        ax.plot(data_do[:nstop, 4], label="enable")
        ax.plot([])
        ax.legend()
        ax.set_xlabel("time step")

    # #########################
    # create DAQmx task
    # #########################
    taskDO = daq.Task()
    taskDO.CreateDOChan("/Dev1/port0/line0:7", "", daq.DAQmx_Val_ChanForAllLines)

    ## Configure timing (from DI task)
    # taskDO.CfgSampClkTiming("OnBoardClock", daq_sample_rate_hz, daq.DAQmx_Val_Rising, daq.DAQmx_Val_ContSamps, samples_per_ch)
    taskDO.CfgSampClkTiming("/Dev1/do/SampleClockTimebase", daq_sample_rate_hz, daq.DAQmx_Val_Rising, daq.DAQmx_Val_ContSamps, samples_per_ch)
    # taskDO.CfgSampClkTiming("OnBoardClock", daq_sample_rate_hz, daq.DAQmx_Val_Rising, daq.DAQmx_Val_FiniteSamps, samples_per_ch)

    ## Write the output waveform
    print("programming DAQ with %d lines" % data_do.shape[0])
    samples_per_ch_ct_digital = ct.c_int32()
    taskDO.WriteDigitalLines(samples_per_ch, False, 10.0, daq.DAQmx_Val_GroupByChannel, data_do,
                             ct.byref(samples_per_ch_ct_digital), None)

    print("DAQ programming complete, elapsed time = %0.2fs" % (time.perf_counter() - tstart))

    # get strings of all available devices
    devs_v = mmc.get_loaded_devices()
    devs = [devs_v.get(ii) for ii in range(devs_v.size())]

    # test sequencing
    # daq = devs[-11]
    # mmc.set_property(daq, "TriggerInputLine", "/Dev1/do/SampleClockTimebase")
    # mmc.load_stage_sequence(daq, [0., 1., 0.])


    # get properties for devices
    # prop_v = mmc.get_device_property_names(devs[5])
    # props = [prop_v.get(ii) for ii in range(prop_v.size())]

    # get allowed values for property
    # prop_vals_v = mmc.get_allowed_property_values(devs[5], props[1])
    # prop_vals = [prop_vals_v.get(ii) for ii in range(prop_vals_v.size())]

    # set a value
    # mmc.set_property("TriggerScope-DAC01", "Volts", 5.1)

    # ensure triggerscope DMD lines are OFF, so OR circuit only cares about NI DAQ
    tscope_dmd_ttl_advance = devs[32] # DMD trigger in line #1 = TTL #7
    mmc.set_property(tscope_dmd_ttl_advance, "State", 0)

    tscope_dmd_ttl_enable = devs[11] # DMD trigger in line #2 = TTL #6
    mmc.set_property(tscope_dmd_ttl_enable, "State", 0)

    # set up camera
    # odt_cam = devs[55]
    odt_cam = devs[1]
    mmc.set_camera_device(odt_cam)
    mmc.set_property(odt_cam, "ScanMode", "2")
    # set external triggering
    mmc.set_property(odt_cam, "TRIGGER ACTIVE", "EDGE")
    mmc.set_property(odt_cam, "TRIGGER DELAY", "0.000")
    # mmc.set_property(odt_cam, "TRIGGER GLOBAL EXPOSURE", "DELAYED")
    mmc.set_property(odt_cam, "TRIGGER GLOBAL EXPOSURE", "GLOBAL RESET")
    mmc.set_property(odt_cam, "TRIGGER SOURCE", "EXTERNAL")
    mmc.set_property(odt_cam, "TriggerPolarity", "POSITIVE")

    # set output signal
    # line 1 trigger ready
    # mmc.set_property(odt_cam, "OUTPUT TRIGGER KIND[0]", "TRIGGER READY")
    mmc.set_property(odt_cam, "OUTPUT TRIGGER KIND[0]", "EXPOSURE")
    mmc.set_property(odt_cam, "OUTPUT TRIGGER POLARITY[0]", "POSITIVE")
    # line 2 at end of readout
    mmc.set_property(odt_cam, "OUTPUT TRIGGER DELAY[1]", "0.0000")
    mmc.set_property(odt_cam, "OUTPUT TRIGGER KIND[1]", "PROGRAMABLE")
    mmc.set_property(odt_cam, "OUTPUT TRIGGER PERIOD[1]", "0.001")
    mmc.set_property(odt_cam, "OUTPUT TRIGGER POLARITY[1]", "POSITIVE")
    mmc.set_property(odt_cam, "OUTPUT TRIGGER SOURCE[1]", "READOUT END")
    # line 3 at start of readout
    mmc.set_property(odt_cam, "OUTPUT TRIGGER DELAY[2]", "0.0000")
    mmc.set_property(odt_cam, "OUTPUT TRIGGER KIND[2]", "PROGRAMABLE")
    mmc.set_property(odt_cam, "OUTPUT TRIGGER PERIOD[2]", "0.001")
    mmc.set_property(odt_cam, "OUTPUT TRIGGER POLARITY[2]", "POSITIVE")
    mmc.set_property(odt_cam, "OUTPUT TRIGGER SOURCE[2]", "VSYNC")
    # set roi
    # roi_rect = bridge.construct_java_object("java.awt.Rectangle", args=[512, 512, 1280, 1280])
    sx = 801
    cx = 1024
    sy = 511
    # cy = 1024
    cy = 1120
    mmc.set_roi(cx - sx//2, cy - sy//2, sx, sy)
    #mm.set_roi()
    # set camera exposure time
    mmc.set_exposure(exposure_time / 1e-3)

    # odt_cam = devs[41]
    # mmc.set_camera_device(odt_cam)
    # # set camera exposure time
    # mmc.set_exposure(exposure_time / 1e-3)
    #
    # # set up trigger in
    # mmc.set_property(odt_cam, "Trigger Mode", "On")
    # mmc.set_property(odt_cam, "Trigger Source", "Line3")
    # mmc.set_property(odt_cam, "Trigger Selector", "FrameStart")
    # mmc.set_property(odt_cam, "Trigger Overlap", "ReadOut")
    # # setup trigger out
    # mmc.set_property(odt_cam, "Line Selector", "Line2")
    # mmc.set_property(bfly, "Line Mode", "Output") # this throws error when call this script from beanshell script in script ...

    # create datastore
    store = mm.data().create_multipage_tiff_datastore(save_dir, True, False)
    mm.displays().manage(store)

    # metadata
    now = datetime.datetime.now()
    now_date = '%04d-%02d-%02d;%02d;%02d;%02d.%03d' % (now.year, now.month, now.day, now.hour, now.minute, now.second, 0)

    dims = mm.data().get_coords_builder().z(npatterns).stage_position(0).time(ntimes).channel(0).build()
    axis_order = ["position", "time", "z", "channel"]

    # pmap = mm.data().get_property_map_builder().put_int("NumPatterns", npatterns).build()
    pmap = mm.data().get_property_map_builder().build()

    summary_mdb = store.get_summary_metadata().copy()
    summary_md = summary_mdb.start_date(now_date).intended_dimensions(dims).user_data(pmap).build()
    # summary_mdb = summary_mdb.intended_dimensions(dims).axis_order(axis_order).user_data(pmap).build()

    store.set_summary_metadata(summary_md)

    # start burst acquisition
    mmc.start_sequence_acquisition(npatterns * ntimes, 0, True)
    # mmc.start_sequence_acquisition(npatterns * ntimes, 0, False)

    # start hardware sequence on DAQ
    taskDO.StartTask()

    start_acquisition = time.perf_counter()
    timeout = 2 * dt * ntimes * nsteps_sequence

    try:
        # collect images
        for tt in range(ntimes):
            for ii in range(npatterns):
                while mmc.get_remaining_image_count() == 0:
                    if time.perf_counter() - start_acquisition > timeout:
                        break

                # pop image
                img_tg = mmc.pop_next_tagged_image()
                img = mm.data().convert_tagged_image(img_tg)
                # # get current md and add to it
                # # img_pm = mm.data().get_property_map_builder().put_int("odt_index", ii)
                # img_md = img.get_metadata().copy().exposure_ms(exposure_time).user_data(img_pm).build()
                # img_md = img.get_metadata().copy().exposure_ms(exposure_time).build()
                # todo: store info about DMD image
                img_md = img.get_metadata().copy().build()
                # get current coords
                img_coords = mm.data().get_coords_builder().stage_position(0).z(ii).channel(0).time(tt).build()
                #
                # # put image in store with metadata and coords
                store.put_image(img.copy_with(img_coords, img_md))

                # store.put_image(img)
    except Exception as e:
        print("error at pattern %d, time point %d" % (ii, tt))
        print(e)
    finally:
        # taskDO.WaitUntilTaskDone(5)
        mmc.stop_sequence_acquisition()

    store.freeze()
    store.close()

try:
    taskDO.StopTask()
    taskDO.ClearTask()
except err:
    print(err)

print("acquisition complete, elapsed time = %0.2fs" % (time.perf_counter() - tstart))

# #########################
# save pattern info
# #########################
pattern_log_fname = os.path.join(save_dir, "pattern_data.pkl")
with open(pattern_log_fname, "wb") as f:
    pickle.dump(pattern_info, f)

# #########################
# turn off lines at end
# #########################
taskDO = daq.Task()
taskDO.CreateDOChan("/Dev1/port0/line0:7", "", daq.DAQmx_Val_ChanForAllLines)
taskDO.WriteDigitalLines(1, 1, 10.0, daq.DAQmx_Val_GroupByChannel,
                         np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8), None, None)

taskDO.StopTask()
taskDO.ClearTask()

print("reset DAQ lines, elapsed time = %0.2fs" % (time.perf_counter() - tstart))
