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
import matplotlib.pyplot as plt
# import pycromanager as pm
import pycromanager as pm
import tifffile
sys.path.append(r"C:\Users\ptbrown2\Desktop\mcsim_private\mcSIM\expt_ctrl")
import dlp6500

# save_dir = r"F:\2021_08_04\pupil_map"

taskDO = daq.Task()
taskDO.CreateDOChan("/Dev1/port0/line0:7", "", daq.DAQmx_Val_ChanForAllLines)
taskDO.WriteDigitalLines(1, 1, 10.0, daq.DAQmx_Val_GroupByChannel,
                         np.array([0, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8), None, None)

taskDO.StopTask()
taskDO.ClearTask()

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

frq = np.array([0, 1/3])
rad = 5
pattern_base = np.round(np.cos(2 * np.pi * (xx * frq[0] + yy * frq[1])), 12)
pattern_base[pattern_base <= 0] = 0
pattern_base[pattern_base > 0] = 1
pattern_base = 1 - pattern_base

# center positions
offs = np.arange(-180, 181, 20)
# offs = np.arange(-45, 46, 5)
# offs = np.arange(-18, 19, 2)
xoffs, yoffs = np.meshgrid(offs, offs)
npatterns = xoffs.size

patterns = np.ones((npatterns, ny, nx), dtype=np.uint8)
for ii in range(npatterns):
    patterns[ii] = np.copy(pattern_base)
    patterns[ii, np.sqrt((xx - cref[1] - xoffs.ravel()[ii])**2 +
                         (yy - cref[0] - yoffs.ravel()[ii])**2) > rad] = 1

tstart = time.process_time()
img_inds, bit_inds = dmd.upload_pattern_sequence(patterns, 105, 0, triggered=True, clear_pattern_after_trigger=False)
# dmd.set_pattern_sequence(img_inds, bit_inds, 105, 0, triggered=True, clear_pattern_after_trigger=True)
tend = time.process_time()
print("loading %d patterns took %0.2fs" % (npatterns, tend - tstart))


# #########################
# program digital output
# #########################
dmd_delay = 105e-6
dt = dmd_delay
DAQ_sample_rate_hz = 1 / dt

n_dmd_pre_trigger = int(np.round(dmd_delay / dt))

# npatterns = 2
exposure_time_est = 0.33e-3
min_frame_time = 3e-3

if exposure_time_est >= min_frame_time:
    nsteps_exposure = int(np.round(exposure_time_est / dt))
    exposure_time = nsteps_exposure * dt
else:
    nsteps_exposure = int(np.ceil(min_frame_time / dt))
    exposure_time = exposure_time_est

# add one extra time step for readout time
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

# #########################
# pycromanager
# #########################
with pm.Bridge() as bridge:
    mmc = bridge.get_core()
    mm = bridge.get_studio()

    devs_v = mmc.get_loaded_devices()
    devs = [devs_v.get(ii) for ii in range(devs_v.size())]
    bfly = devs[41]

    mmc.set_camera_device(bfly)
    # set camera exposure time
    mmc.set_exposure(exposure_time / 1e-3)
    # mmc.getDevicePropertyNames()
    # mmc.getAllowedPropertyValues()
    # set up trigger in
    mmc.set_property(bfly, "Trigger Mode", "On")
    mmc.set_property(bfly, "Trigger Source", "Line3")
    mmc.set_property(bfly, "Trigger Selector", "FrameStart")
    mmc.set_property(bfly, "Trigger Overlap", "ReadOut")
    # setup trigger out
    mmc.set_property(bfly, "Line Selector", "Line2")

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

    try:
        # collect images
        for ii in range(npatterns):
            while mmc.get_remaining_image_count() == 0:
                pass

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
        print(e)
    finally:
        taskDO.WaitUntilTaskDone(5)
        mmc.stop_sequence_acquisition()

    store.freeze()
    store.close()

taskDO.StopTask()
taskDO.ClearTask()

# #########################
# turn off lines at end
# #########################
taskDO = daq.Task()
taskDO.CreateDOChan("/Dev1/port0/line0:7", "", daq.DAQmx_Val_ChanForAllLines)
taskDO.WriteDigitalLines(1, 1, 10.0, daq.DAQmx_Val_GroupByChannel,
                         np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8), None, None)

taskDO.StopTask()
taskDO.ClearTask()
