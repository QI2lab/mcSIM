import numpy as np

# DAQ line mapping
daq_do_map = {"odt_cam": 0,
              "odt_shutter": 1,
              "odt_laser": 2,
              "sim_cam": 5,
              "sim_shutter": 6,
              "blue_laser": 7,
              "green_laser": 8,
              "red_laser": 9,
              "dmd_enable": 4,
              "dmd_advance": 3}

# get line names
k = list(daq_do_map.keys())
v = np.array(list(daq_do_map.values()))
ind_max = np.max(v) + 1

do_line_names = []
for ii in range(ind_max):
    ind = np.where(v == ii)[0]
    if ind.size == 1:
        ind = int(ind)
        name = k[ind]
    else:
        name = ""
    do_line_names.append(name)

# analog lines
daq_ao_map = {"vc_mirror_x": 0,
              "vc_mirror_y": 1,
              "z_stage": 2}

# dictionary of presets
# each preset is a dictionary with keys "digital" and "analog"
# preset["blue"]["digital"] is a dictionary where the keys are the channel names
# todo: should I also add the DMD pictures as an argument here?
presets = {"off": {"digital": {"odt_cam": 0, "odt_shutter": 0, "odt_laser": 0,
                                "sim_cam": 0, "sim_shutter": 0,
                                "blue_laser": 0, "green_laser": 0, "red_laser": 0,
                                "dmd_enable": 0, "dmd_advance": 0},
                    "analog": {"vc_mirror_x": 0, "vc_mirror_y": 0}},
           "blue": {"digital": {"odt_cam": 0, "odt_shutter": 0, "odt_laser": 0,
                                "sim_cam": 0, "sim_shutter": 1,
                                "blue_laser": 1, "green_laser": 0, "red_laser": 0,
                                "dmd_enable": 1, "dmd_advance": 1},
                    "analog": {"vc_mirror_x": 0, "vc_mirror_y": 0}},
           "green": {"digital": {"odt_cam": 0, "odt_shutter": 0, "odt_laser": 0,
                                 "sim_cam": 0, "sim_shutter": 1,
                                 "blue_laser": 0, "green_laser": 0, "red_laser": 1,
                                 "dmd_enable": 1, "dmd_advance": 1},
                     "analog": {"vc_mirror_x": 0, "vc_mirror_y": 0}},
           "red": {"digital": {"odt_cam": 0, "odt_shutter": 0, "odt_laser": 0,
                               "sim_cam": 0, "sim_shutter": 1,
                               "blue_laser": 0, "green_laser": 0, "red_laser": 1,
                               "dmd_enable": 1, "dmd_advance": 1},
                   "analog": {"vc_mirror_x": 0, "vc_mirror_y": 0}},
           "odt": {"digital": {"odt_cam": 0, "odt_shutter": 1, "odt_laser": 1,
                               "sim_cam": 0, "sim_shutter": 0,
                               "blue_laser": 0, "green_laser": 0, "red_laser": 0,
                               "dmd_enable": 1, "dmd_advance": 1},
                   "analog": {}}
           }

def preset_to_array(preset, do_line_names, nchannels=None):
    if nchannels is None:
        nchannels = len(preset["digital"])

    digital_array = np.zeros((nchannels), dtype=np.uint8)
    for ii in range(len(preset["digital"])):
        digital_array[ii] = preset["digital"][do_line_names[ii]]

    return digital_array