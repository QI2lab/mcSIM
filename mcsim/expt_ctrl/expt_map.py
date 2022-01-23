import numpy as np

# DAQ line mapping
daq_do_map = {"odt_cam": 0,
              "odt_shutter": 1,
              "odt_laser": 2,
              "sim_cam": 5,
              "sim_shutter": 6,
              "blue_laser": 8,
              "green_laser": 7,
              "red_laser": 9,
              "dmd_enable": 4,
              "dmd_advance": 3}

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
                                 "blue_laser": 0, "green_laser": 1, "red_laser": 0,
                                 "dmd_enable": 1, "dmd_advance": 1},
                     "analog": {"vc_mirror_x": -4.75, "vc_mirror_y": -1.05}},
           "red": {"digital": {"odt_cam": 0, "odt_shutter": 0, "odt_laser": 0,
                               "sim_cam": 0, "sim_shutter": 1,
                               "blue_laser": 0, "green_laser": 0, "red_laser": 1,
                               "dmd_enable": 1, "dmd_advance": 1},
                   "analog": {"vc_mirror_x": 0, "vc_mirror_y": 0}},
           "odt": {"digital": {"odt_cam": 0, "odt_shutter": 1, "odt_laser": 1,
                               "sim_cam": 0, "sim_shutter": 0,
                               "blue_laser": 0, "green_laser": 0, "red_laser": 0,
                               "dmd_enable": 1, "dmd_advance": 1},
                   "analog": {"vc_mirror_x": 0, "vc_mirror_y": 0}}
           }

def get_line_names(map):
    """
    From a dictionary of line names, get list of line names such that line ii is called name[ii]
    """
    # get line names
    k = list(map.keys())
    v = np.array(list(map.values()))
    ind_max = np.max(v)

    nchannels = ind_max + 1

    line_names = []
    for ii in range(nchannels):

        # todo: catch case where we get multiple matches
        ind = np.where(v == ii)[0]
        if ind.size == 1:
            ind = int(ind)
            name = k[ind]
        else:
            name = ""
        line_names.append(name)

    return line_names


def preset_to_array(preset, do_map, ao_map, n_digital_channels=None, n_analog_channels=None):
    """
    Get arrays to program daq from prese
    """

    # get digital array
    if n_digital_channels is None:
        n_digital_channels = max(list(do_map.values())) + 1

    digital_array = np.zeros((n_digital_channels), dtype=np.uint8)
    for name in list(preset["digital"].keys()):
        digital_array[do_map[name]] = preset["digital"][name]

    # get analog array
    if n_analog_channels is None:
        n_analog_channels = max(list(ao_map.values())) + 1

    analog_array = np.zeros((n_analog_channels))
    for name in list(preset["analog"].keys()):
        analog_array[ao_map[name]] = preset["analog"][name]


    return digital_array, analog_array
