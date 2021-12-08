"""
Script intended to be called from MM beanshell

In beanshell, I remove the NI devices, then run this script, then afterwards re-add the NI devices.
This is a workaround to use MM for cruising around the sample but PyDaqMx for controlling the DAQ
"""
import time
import os
import sys
import datetime
import PyDAQmx as daq
import ctypes as ct
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pycromanager as pm
import tifffile
# todo: do I need this? If so should add relative path
# sys.path.append(r"C:\Users\ptbrown2\Desktop\mcsim_private\mcSIM\expt_ctrl")
import dlp6500


tstart = time.perf_counter()
# save_dir = r"F:\2021_08_04\pupil_map"

# use this if device won't restart
# daq.DAQmxResetDevice("Dev1")

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

# ang = 20 * np.pi/180
# ang = 0 * np.pi/180
# frq = np.array([np.sin(ang), np.cos(ang)]) * 1/3
ang = -45 * np.pi/180
frq = np.array([np.sin(ang), np.cos(ang)]) * 1/4 * np.sqrt(2)

rad = 5
# rad = 10
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
n_phis = 10
phis = np.arange(n_phis) * 2*np.pi / n_phis
fractions = [0.25, 0.5, 0.75, 0.9]
n_thetas = len(fractions)

xoffs = np.zeros((n_phis, n_thetas))
yoffs = np.zeros((n_phis, n_thetas))
for ii in range(n_phis):
    for jj in range(n_thetas):
        xoffs[ii, jj] = np.cos(phis[ii]) * pupil_rad_mirrors * fractions[jj]
        yoffs[ii, jj] = np.sin(phis[ii]) * pupil_rad_mirrors * fractions[jj]

xoffs = np.concatenate((np.array([0]), xoffs.ravel()))
yoffs = np.concatenate((np.array([0]), yoffs.ravel()))

# offs = np.arange(-360, 400, 40)
# offs = np.arange(-180, 181, 20)
# offs = np.arange(-45, 46, 5)
# offs = np.arange(-18, 19, 2)
# offs = np.zeros(19)
# xoffs, yoffs = np.meshgrid(offs, offs)

npatterns = xoffs.size

pattern_info = {"frquency": frq, "radius": rad, "cref": cref, "phase": phase, "dmd_size": dmd_size,
                "xoffsets": xoffs, "yoffsets": yoffs, "npatterns": npatterns}

print("generating patterns...")
if True:
    patterns = np.ones((npatterns, ny, nx), dtype=np.uint8)
    for ii in range(npatterns):
        patterns[ii] = np.copy(pattern_base)
        patterns[ii, np.sqrt((xx - cref[1] - xoffs.ravel()[ii])**2 +
                             (yy - cref[0] - yoffs.ravel()[ii])**2) > rad] = 1
else:
    # test repeates of all ON
    patterns = np.ones((npatterns, ny, nx), dtype=np.uint8)

print("finished generating patterns, elapsed time %0.2fs" % (time.perf_counter() - tstart))


img_inds, bit_inds = dmd.upload_pattern_sequence(patterns, 105, 0, triggered=True, clear_pattern_after_trigger=False)
# dmd.set_pattern_sequence(img_inds, bit_inds, 105, 0, triggered=True, clear_pattern_after_trigger=True)
print("loaded %d patterns, elapsed time %0.2fs" % (npatterns, time.perf_counter() - tstart))

# #########################
# program digital output
# #########################
dmd_delay = 105e-6
dt = dmd_delay
DAQ_sample_rate_hz = 1 / dt

n_dmd_pre_trigger = int(np.round(dmd_delay / dt))

exposure_time_est = 2.8e-3
# exposure_time_est = 1e-3
# min_frame_time = 3e-3
min_frame_time = 30e-3

if exposure_time_est >= min_frame_time:
    nsteps_exposure = int(np.round(exposure_time_est / dt))
    exposure_time = nsteps_exposure * dt
else:
    nsteps_exposure = int(np.ceil(min_frame_time / dt))
    exposure_time = exposure_time_est
print("exposure time = %0.2fms" % (exposure_time * 1e3))

# add extra time steps for readout time
nsteps_pattern = nsteps_exposure + 3

# add extra steps at start to turn on laser and DMD enable trigger and allow to stabilize
# alternatively, could set this beforehand and not be in sequence...
stabilize_time = 5e-3
nstabilize = int(np.ceil(stabilize_time / dt))
# second, trigger DMD one time step early because it has a 105us delay
samples_per_ch = nsteps_pattern * npatterns + nstabilize
data_do = np.zeros((samples_per_ch, 8), dtype=np.uint8)

# set camera trigger
# for ii in range(nsteps_exposure):
#     data_do[ii::nsteps_pattern, 0] = 1
data_do[nstabilize::nsteps_pattern, 0] = 1

# shutter
data_do[:, 1] = 1

# laser
data_do[:, 2] = 1

# DMD enable trigger
data_do[:, 4] = 1

# DMD advance trigger
# trigger on step before camera to account for 105us DMD delay
data_do[nstabilize - n_dmd_pre_trigger:-1:nsteps_pattern, 3] = 1

# print(data_do)

taskDO = daq.Task()
taskDO.CreateDOChan("/Dev1/port0/line0:7", "", daq.DAQmx_Val_ChanForAllLines)

## Configure timing (from DI task)
# taskDO.CfgSampClkTiming("OnBoardClock", DAQ_sample_rate_hz, daq.DAQmx_Val_Rising, daq.DAQmx_Val_ContSamps, samples_per_ch)
taskDO.CfgSampClkTiming("OnBoardClock", DAQ_sample_rate_hz, daq.DAQmx_Val_Rising, daq.DAQmx_Val_FiniteSamps, samples_per_ch)

## Write the output waveform
print("programming DAQ with %d lines" % data_do.shape[0])
samples_per_ch_ct_digital = ct.c_int32()
taskDO.WriteDigitalLines(samples_per_ch, False, 10.0, daq.DAQmx_Val_GroupByChannel, data_do,
                         ct.byref(samples_per_ch_ct_digital), None)

print("DAQ programming complete, elapsed time = %0.2fs" % (time.perf_counter() - tstart))
# #########################
# pycromanager
# #########################
with pm.Bridge() as bridge:
    mmc = bridge.get_core()
    mm = bridge.get_studio()

    # get strings of all available devices
    devs_v = mmc.get_loaded_devices()
    devs = [devs_v.get(ii) for ii in range(devs_v.size())]

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
    # set external triggering
    mmc.set_property(odt_cam, "TRIGGER SOURCE", "EXTERNAL")
    mmc.set_property(odt_cam, "TriggerPolarity", "POSITIVE")
    # set roi
    # roi_rect = bridge.construct_java_object("java.awt.Rectangle", args=[512, 512, 1280, 1280])
    sx = 801
    cx = 1024
    sy = 511
    cy = 1024
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

    # pull save dir from MDA
    acq_settings = mm.get_acquisition_manager().get_acquisition_settings()
    save_dir = os.path.join(acq_settings.root(), acq_settings.prefix())
    save_dir = mm.data().get_unique_save_directory(save_dir)

    # create datastore
    store = mm.data().create_multipage_tiff_datastore(save_dir, True, False)
    mm.displays().manage(store)

    # metadata
    now = datetime.datetime.now()
    now_date = '%04d-%02d-%02d;%02d;%02d;%02d.%03d' % (now.year, now.month, now.day, now.hour, now.minute, now.second, 0)

    dims = mm.data().get_coords_builder().z(0).stage_position(0).time(0).channel(0).build()
    axis_order = ["position", "time", "z", "channel"]

    # pmap = mm.data().get_property_map_builder().put_int("NumPatterns", npatterns).build()
    pmap = mm.data().get_property_map_builder().build()

    summary_mdb = store.get_summary_metadata().copy()
    summary_md = summary_mdb.start_date(now_date).intended_dimensions(dims).user_data(pmap).build()
    # summary_mdb = summary_mdb.intended_dimensions(dims).axis_order(axis_order).user_data(pmap).build()

    store.set_summary_metadata(summary_md)

    # start burst acquisition
    mmc.start_sequence_acquisition(npatterns, 0, True)

    # start hardware sequence on DAQ
    taskDO.StartTask()

    start_acquisition = time.perf_counter()
    timeout = 30
    try:
        # collect images
        for ii in range(npatterns):
            while mmc.get_remaining_image_count() == 0:
                if time.perf_counter() - start_acquisition > 3:
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
            img_coords = mm.data().get_coords_builder().stage_position(0).z(0).channel(0).time(ii).build()
            #
            # # put image in store with metadata and coords
            store.put_image(img.copy_with(img_coords, img_md))
            # store.put_image(img)
    except Exception as e:
        print("error at image %d" % ii)
        print(e)
    finally:
        taskDO.WaitUntilTaskDone(5)
        mmc.stop_sequence_acquisition()

    store.freeze()
    store.close()

taskDO.StopTask()
taskDO.ClearTask()

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
