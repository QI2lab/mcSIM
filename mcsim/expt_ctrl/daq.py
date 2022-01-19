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
    def __init__(self, dev_name="Dev1", digital_lines="port0/line0:15", analog_lines=["ao0", "ao1", "ao2"]):

        self.dev_name = dev_name
        self.digital_lines = "/" + dev_name + "/" + digital_lines
        self.n_digital_lines = 16
        self.analog_lines = ["/" + dev_name + "/" + line for line in analog_lines]
        self.n_analog_lines = 3

        self.task_do = None
        self.task_ao = None

    def reset(self):
        daqmx.DAQmxResetDevice(self.dev_name)
        self.set_digital_once(np.zeros(self.n_digital_lines))

    def set_digital_once(self, array):
        array = np.array(array).astype(np.uint8)
        if array.ndim != 1 or array.size != self.n_digital_lines:
            raise ValueError()

        self.task_do = daqmx.Task()
        self.task_do.CreateDOChan(self.digital_lines, "", daqmx.DAQmx_Val_ChanForAllLines)
        self.task_do.WriteDigitalLines(1, 1, 10.0, daqmx.DAQmx_Val_GroupByChannel, array, None, None)
        self.task_do.StopTask()
        self.task_do.ClearTask()

    def set_analog_once(self, array):
        array = np.array(array)
        # if array.ndim != 1 or array.size != self.n_digital_lines:
        #     raise ValueError()

        self.task_ao = daqmx.Task()
        self.task_ao.CreateAOVoltageChan(self.analog_lines, "", -6.0, 6.0, daqmx.DAQmx_Val_Volts, None)
        self.task_ao.WriteAnalogScalarF64(True, -1, array, None)
        self.task_ao.StopTask()
        self.task_ao.ClearTask()

    def set_sequence(self, digital_array, analog_array, sample_rate_hz, clock_source="OnBoardClock"):
        if not digital_array.shape[0] == analog_array.shape[0]:
            raise ValueError("digital_array and analog_array should have the same size in their first dimension")

        samples_per_ch = digital_array.shape[0]

        # ######################
        # digital task
        # ######################
        self.task_do = daqmx.Task()
        self.task_do.CreateDOChan(self.digital_lines, "", daqmx.DAQmx_Val_ChanForAllLines)

        # Configure timing (from DI task)
        self.task_do.CfgSampClkTiming(clock_source, sample_rate_hz, daqmx.DAQmx_Val_Rising, daqmx.DAQmx_Val_ContSamps, samples_per_ch)

        # self.task_do.ExportSignal(clock_source, "/Dev1/PFI2")

        # Write the output waveform
        print("programming DAQ with %d lines" % digital_array.shape[0])
        samples_per_ch_ct_digital = ct.c_int32()
        self.task_do.WriteDigitalLines(samples_per_ch, False, 10.0, daqmx.DAQmx_Val_GroupByChannel, digital_array,
                                       ct.byref(samples_per_ch_ct_digital), None)

        # ######################
        # analog task
        # ######################
        # todo: maybe could only trigger every n digital clock cycles?
        # todo: or trigger on digital line!

        self.task_ao = daqmx.Task()
        self.task_ao.CreateAOVoltageChan(", ".join(self.analog_lines), "", -5.0, 5.0, daqmx.DAQmx_Val_Volts, None)

        self.task_ao.CfgSampClkTiming("", sample_rate_hz, daqmx.DAQmx_Val_Rising, daqmx.DAQmx_Val_ContSamps,
                                          samples_per_ch)
        self.task_ao.CfgDigEdgeStartTrig("/Dev1/do/StartTrigger", daqmx.DAQmx_Val_Rising)

        samples_per_ch_ct = ct.c_int32()
        self.task_ao.WriteAnalogF64(samples_per_ch, False, 10.0, daqmx.DAQmx_Val_GroupByScanNumber,
                                            analog_array, ct.byref(samples_per_ch_ct), None)



    def start_sequence(self):
        self.task_ao.StartTask()
        self.task_do.StartTask()


    def stop_sequence(self):
        self.task_do.StopTask()
        self.task_do.ClearTask()

        self.task_ao.StopTask()
        self.task_ao.ClearTask()

def plot_arr(arr, line_names=None, title=""):
    """
    Plot daq array
    @param arr: nsteps x nchannels array
    @param title:
    @return:
    """

    if line_names is None:
        line_names = [str(ii) for ii in range(arr.shape[1])]

    ind_max = len(line_names) + 1
    ticks = []
    for ii in range(ind_max):
        ticks.append(matplotlib.text.Text(float(ii), 0, line_names[ii]))


    figh = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title(title)

    ax.imshow(arr[:, :ind_max], aspect="auto", interpolation="none")

    ax.set_xticks(range(ind_max))
    ax.set_xticklabels(ticks)
    ax.set_xlabel("channel")
    ax.set_ylabel("time step")

    return figh

if __name__ == "__main__":
    nt = 10
    sample_rate_hz = 1 / 1e-3

    do = np.zeros((nt, 16), dtype=np.uint8)
    do[::2, 0] = 1

    ao = np.zeros((nt, 3))
    ao[::2, 0] = 3.5
    ao[1::2, 1] = 3.75
    ao[::2, 2] = 2.75

    d = nidaq()
    d.set_sequence(do, ao, sample_rate_hz)
    d.start_sequence()