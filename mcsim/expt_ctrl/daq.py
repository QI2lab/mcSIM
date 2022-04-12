"""
Code for controlling data acquisition cards (DAQ).

Different models of DAQ card should inherit from the class daq (think of this like a java interface).

todo: could split different implementations to different files to avoid dependency issues
"""
import numpy as np
import PyDAQmx as daqmx
import matplotlib.pyplot as plt
import matplotlib
import ctypes as ct

class daq():
    def __init__(self):
        pass

    def set_digital_once(self, array):
        pass

    def set_analog_once(self, array):
        pass

    def set_sequence(self):
        pass

    def start_sequence(self):
        pass

    def stop_sequence(self):
        pass


class nidaq(daq):
    """
    Class for controlling National Instruments DAQ
    # todo: add ability to set device by name?
    # todo: want better way to deal with digital line addresses. Maybe should store these for each line in self.digital_lines
    # todo: and separately store the block indices somehow...
    """

    def __init__(self, dev_name: str = "Dev1", digital_lines: str = "port0/line0:15", analog_lines: list = ["ao0", "ao1", "ao2"],
                 digital_line_names: dict = None, analog_line_names: dict = None):
        """

        @param dev_name: device names, typically of the form `Devk` for k an integer
        @param digital_lines:
        @param analog_lines: list of analog lines
        @param digital_line_names: dictionary where keys give the name of the lines and values give the line index.
        It is not necessary for every line to have a name
        @param analog_line_names:
        """

        self.dev_name = dev_name

        # digital lines
        self.digital_lines = "/" + dev_name + "/" + digital_lines
        self.n_digital_lines = 16
        self.digital_lines_addresses = [f"{dev_name:s}/port0/line{ii:d}" for ii in range(16)]
        self.digital_line_names = digital_line_names

        # analog lines
        # self.analog_lines = ["/" + dev_name + "/" + line for line in analog_lines]
        self.analog_lines = [f"/{dev_name}/{line}" for line in analog_lines]
        self.n_analog_lines = len(self.analog_lines)
        self.analog_line_names = analog_line_names

        # task handles. generally do not access directly
        self._task_do = None
        self._task_ao = None


    def reset(self):
        """

        @return:
        """
        daqmx.DAQmxResetDevice(self.dev_name)
        self.set_digital_once(np.zeros(self.n_digital_lines))


    def set_digital_once(self, array):
        """
        Set digital lines as a block

        @param array: sizes self.n_digital_lines
        @return:
        """
        array = np.array(array).astype(np.uint8)
        if array.ndim != 1 or array.size != self.n_digital_lines:
            raise ValueError(f"array must have shape {self.n_digital_lines:d} but had shape {array.shape}")

        self._task_do = daqmx.Task()
        self._task_do.CreateDOChan(self.digital_lines, "", daqmx.DAQmx_Val_ChanForAllLines)
        self._task_do.WriteDigitalLines(1, 1, 10.0, daqmx.DAQmx_Val_GroupByChannel, array, None, None)
        self._task_do.StopTask()
        self._task_do.ClearTask()


    def set_digital_lines(self, array: np.ndarray, lines: list = None):
        """
        Set digital lines

        @param array: 1D array of same size as number of lines you want to set
        @param lines: index of lines to set
        @return:
        """
        if lines is None:
            lines = list(range(self.n_digital_lines))

        if array.shape != (len(lines),):
            raise ValueError(f"array must have shape {len(lines):d}, but had shape {array.shape}")

        for ii, l in enumerate(lines):
            task_do = daqmx.Task()
            task_do.CreateDOChan(self.digital_lines_addresses[l], "", daqmx.DAQmx_Val_ChanForAllLines)
            task_do.WriteDigitalLines(1, 1, 10.0, daqmx.DAQmx_Val_GroupByChannel, np.atleast_1d(array[ii]).astype(np.uint8), None, None)
            task_do.StopTask()
            task_do.ClearTask()


    def set_digital_lines_by_name(self, array: np.ndarray, line_names: list):
        if self.digital_line_names is None:
            raise ValueError("cannot set lines by name because self.digital_line_names is None")
        lines = [self.digital_line_names[n] for n in line_names]

        return self.set_digital_lines(array, lines)

    def set_analog_once(self, array):
        """

        @param array:
        @return:
        """
        array = np.array(array)
        # if array.ndim != 1 or array.size != self.n_digital_lines:
        #     raise ValueError()

        # can't get WriteAnalogScalarF64() to work with multiple lines, so
        if len(array) != len(self.analog_lines):
            raise ValueError()

        for ii in range(len(self.analog_lines)):
            self._task_ao = daqmx.Task()
            self._task_ao.CreateAOVoltageChan(self.analog_lines[ii], "", -5.0, 5.0, daqmx.DAQmx_Val_Volts, None)
            self._task_ao.WriteAnalogScalarF64(True, daqmx.DAQmx_Val_WaitInfinitely, array[ii], None)
            self._task_ao.StopTask()
            self._task_ao.ClearTask()


    def set_sequence(self, digital_array, analog_array, sample_rate_hz, clock_source="OnBoardClock", continuous=True):
        """

        @param digital_array:
        @param analog_array:
        @param sample_rate_hz:
        @param clock_source:
        @param continuous: if True, then sequence will be run repeatedly. If False, sequence will be run once.
        @return:
        """
        if not digital_array.shape[0] == analog_array.shape[0]:
            raise ValueError("digital_array and analog_array should have the same size in their first dimension")

        samples_per_ch = digital_array.shape[0]

        # ######################
        # digital task
        # ######################
        self._task_do = daqmx.Task()
        self._task_do.CreateDOChan(self.digital_lines, "", daqmx.DAQmx_Val_ChanForAllLines)

        # Configure timing (from DI task)
        if continuous:
            repeat = daqmx.DAQmx_Val_ContSamps
        else:
            repeat = daqmx.DAQmx_Val_FiniteSamps
        self._task_do.CfgSampClkTiming(clock_source, sample_rate_hz, daqmx.DAQmx_Val_Rising, repeat, samples_per_ch)

        # self.task_do.ExportSignal(clock_source, "/Dev1/PFI2")

        # Write the output waveform
        print("programming DAQ with %d lines" % digital_array.shape[0])
        samples_per_ch_ct_digital = ct.c_int32()
        self._task_do.WriteDigitalLines(samples_per_ch, False, 10.0, daqmx.DAQmx_Val_GroupByChannel, digital_array,
                                        ct.byref(samples_per_ch_ct_digital), None)

        # ######################
        # analog task
        # ######################
        # todo: maybe could only trigger every n digital clock cycles?
        # todo: or trigger on digital line!

        self._task_ao = daqmx.Task()
        self._task_ao.CreateAOVoltageChan(", ".join(self.analog_lines), "", -5.0, 5.0, daqmx.DAQmx_Val_Volts, None)

        self._task_ao.CfgSampClkTiming("", sample_rate_hz, daqmx.DAQmx_Val_Rising, daqmx.DAQmx_Val_ContSamps,
                                       samples_per_ch)
        self._task_ao.CfgDigEdgeStartTrig("/Dev1/do/StartTrigger", daqmx.DAQmx_Val_Rising)

        samples_per_ch_ct = ct.c_int32()
        self._task_ao.WriteAnalogF64(samples_per_ch, False, 10.0, daqmx.DAQmx_Val_GroupByScanNumber,
                                     analog_array, ct.byref(samples_per_ch_ct), None)



    def start_sequence(self):
        self._task_ao.StartTask()
        self._task_do.StartTask()


    def stop_sequence(self):
        try:
            self._task_do.StopTask()
            self._task_do.ClearTask()
        except daqmx.DAQmxFunctions.InvalidTaskError:
            pass

        try:
            self._task_ao.StopTask()
            self._task_ao.ClearTask()
        except daqmx.DAQmxFunctions.InvalidTaskError:
            pass


def plot_arr(arr, line_names=None, title="", **kwargs):
    """
    Plot daq arrays
    @param arr: nsteps x nchannels array
    @param title:
    @return:
    """

    if line_names is None:
        line_names = [str(ii) for ii in range(arr.shape[1])]

    ind_max = len(line_names) #+ 1
    ticks = []
    for ii in range(ind_max):
        ticks.append(matplotlib.text.Text(float(ii), 0, line_names[ii]))


    figh = plt.figure(**kwargs)
    ax = figh.add_subplot(1, 1, 1)
    ax.set_title(title)

    ax.imshow(arr[:, :ind_max], aspect="auto", interpolation="none")

    ax.set_xticks(range(ind_max))
    ax.set_xticklabels(ticks)
    ax.set_xlabel("channel")
    ax.set_ylabel("time step")

    return figh
