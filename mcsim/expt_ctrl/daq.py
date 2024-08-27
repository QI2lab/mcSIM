"""
Code for controlling data acquisition cards (DAQ).

Different models of DAQ card should inherit from the class daq (think of this like a java interface).
"""
from typing import Optional
import datetime
from re import match
import json
import ctypes as ct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from warnings import warn

try:
    import PyDAQmx as daqmx
except ImportError:
    daqmx = None
    warn("PyDAQmx could not be imported")


class daq:
    def __init__(self):
        pass

    def set_digital_lines_by_address(self, values, addresses):
        pass

    def set_analog_lines_by_address(self, array, addresses):
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
    # todo: want better way to deal with digital line addresses.
    # todo: Maybe should store these for each line in self.digital_lines
    # todo: and separately store the block indices somehow...

    # todo: right now stores addresses and etc. in lists and use the index of the list as the index of the lines
    # todo: but this could be very confusing e.g. if have port0/0:15 and then some other line like port1/1
    # todo: in that case have 0, ..., 15 representing first lines and then 16 represents port1/1.
    # todo: so maybe better to store dictionary with direct mappings between line names and addresses?
    """

    def __init__(self,
                 dev_name: str = "Dev1",
                 digital_lines: str = "port0/line0:15",
                 analog_lines: tuple = ("ao0", "ao1", "ao2"),
                 analog_input_lines: tuple[str] = ("ai0",),
                 digital_line_names: Optional[dict] = None,
                 analog_line_names: Optional[dict] = None,
                 presets: Optional[dict] = None,
                 config_file: str = None,
                 component: str = Optional[None],
                 initialize: bool = True):
        """
        Initialize DAQ. Note that DAQ can be instantiated before the actual DAQ is present

        :param dev_name: device names, typically of the form `Devk` for k an integer
        :param digital_lines:
        :param analog_lines: list of analog lines
        :param digital_line_names: dictionary where keys give the name of the lines and values give the line index.
          It is not necessary for every line to have a name
        :param analog_line_names:
        :param presets: dictionary of presets
        :param config_file: alternative method of provided digital_line_names, analog_line_names, and presets.
          If config_file is supplied, these other keyword arguments should not be supplied
        :param component: component in json file to load configuration from
        :param initialize:
        """
        super().__init__()

        if config_file is not None and (digital_line_names is not None or
                                        analog_line_names is not None or
                                        presets is not None):
            raise ValueError("config_file and either digital_line_names, analog_line_names, or presets"
                             " were both provided. If config_file is provided, do not also provide this other info")

        if config_file is not None:
            digital_line_names, analog_line_names, presets, _ = load_config_file(config_file, component)

        self.dev_name = dev_name

        # todo: can I read lines from device like this? daqmx.GetDevTerminals()

        # digital output lines
        self.digital_lines = f"/{dev_name}/{digital_lines}"  # todo: get rid of in favor of digital_lines_addresses
        self.n_digital_lines = 16  # todo: want to detect not hard code
        self.digital_lines_addresses = [f"/{dev_name:s}/port0/line{ii:d}" for ii in range(self.n_digital_lines)]
        self.digital_line_names = digital_line_names
        self.last_known_digital_val = np.zeros(self.n_digital_lines, dtype=np.uint8)
        self.do_re = ".*Dev(\d+).*port(\d+).*line(\d+)"

        # analog output lines
        self.analog_lines = [f"/{dev_name}/{line}" for line in analog_lines]
        self.n_analog_lines = len(self.analog_lines)
        self.analog_line_names = analog_line_names
        self.last_known_analog_val = np.zeros(self.n_analog_lines, dtype=float)
        # self.ao_re = ".*Dev(\d+).*ao(\d+)"

        # analog input lines
        self.analog_input_line_names = [f"/{dev_name}/{line}" for line in analog_input_lines]
        self.n_analog_inputs = len(self.analog_input_line_names)

        # preset states
        self.presets = presets

        # task handles
        self._task_do = None
        self._task_ao = None
        self._task_di = None
        self._task_ai = None
        self._task_ct = None

        # any code which requires device to already be present should go inside this block
        self.initialized = initialize
        if self.initialize:
            pass

    def initialize(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        self.__init__(initialize=True, **kwargs)

    def reset(self):
        """
        reset device

        :return:
        """
        daqmx.DAQmxResetDevice(self.dev_name)
        self.set_digital_once(np.zeros(self.n_digital_lines))

    def get_do_address(self, dev, port, line):
        """

        :param dev:
        :param port:
        :param line:
        :return:
        """
        # todo: remove if not useful
        return f"Dev{dev:d}/port{port:d}/line{line:d}"

    def read_do_address(self, address):
        """

        :param address:
        :return:
        """
        # todo: remove if not useful
        m = match(self.do_re, address)

        if m is None:
            raise ValueError("")

        dev = int(m.group(1))
        port = int(m.group(2))
        line = int(m.group(3))

        return dev, port, line

    def get_do_block_address(self, address_list):
        """

        :param address_list:
        :return:
        """
        # todo: still playing with this...remove if not useful

        devs, ports, lines = zip(*[self.read_do_address(ad) for ad in address_list])

        if not all([d == devs[0] for d in devs]):
            raise ValueError()

        if not all([p == ports[0] for p in ports]):
            raise ValueError()

        if not list(lines) == list(range(lines[0], lines[-1] + 1)):
            raise ValueError()

        address_block = f"Dev{devs[0]:d}/port{ports[0]:d}/line{lines[0]}:{lines[-1]}"

        return address_block

    def set_digital_once(self,
                         array: np.ndarray):
        """
        Set digital lines as a block

        :param: sizes self.n_digital_lines
        :return:
        """
        array = np.array(array).astype(np.uint8)
        if array.ndim != 1 or array.size != self.n_digital_lines:
            raise ValueError(f"array must have shape {self.n_digital_lines:d} but had shape {array.shape}")

        self._task_do = daqmx.Task()
        self._task_do.CreateDOChan(self.digital_lines,
                                   "",
                                   daqmx.DAQmx_Val_ChanForAllLines)
        self._task_do.WriteDigitalLines(1,
                                        1,
                                        10.0,
                                        daqmx.DAQmx_Val_GroupByChannel,
                                        array,
                                        None,
                                        None)
        self._task_do.StopTask()
        self._task_do.ClearTask()

        self.last_known_digital_val[:] = array

    def set_digital_lines_by_address(self,
                                     array: np.ndarray,
                                     addresses: list):
        if array.shape != (len(addresses),):
            raise ValueError(f"array must have shape {len(addresses):d}, but had shape {array.shape}")

        try:
            self._task_do = daqmx.Task()
            self._task_do.CreateDOChan(", ".join(addresses),
                                       "",
                                       daqmx.DAQmx_Val_ChanForAllLines)
            self._task_do.WriteDigitalLines(1,
                                            1,
                                            10.0,
                                            daqmx.DAQmx_Val_GroupByChannel,
                                            np.atleast_1d(array).astype(np.uint8),
                                            None,
                                            None)

            # cache values
            for ii, ad in enumerate(addresses):
                ind = np.nonzero([a == ad for a in self.digital_lines_addresses])[0]
                if len(ind) > 0:
                    self.last_known_digital_val[ind] = array[ii]

        # todo: get exception type of trying to write invalid data and only catch that specific type of exception...
        except Exception as e:
            print(e)
        finally:
            self._task_do.StopTask()
            self._task_do.ClearTask()

    def set_digital_lines_by_index(self,
                                   array: np.ndarray,
                                   lines: Optional[list] = None):
        """
        Set digital lines

        :param array: 1D array of same size as number of lines you want to set
        :param lines: index of lines to set
        :return:
        """
        if lines is None:
            lines = list(range(self.n_digital_lines))

        if array.shape != (len(lines),):
            raise ValueError(f"array must have shape {len(lines):d}, but had shape {array.shape}")

        addresses = [self.digital_lines_addresses[l] for l in lines]
        return self.set_digital_lines_by_address(array, addresses)

    def set_digital_lines_by_name(self,
                                  array: np.ndarray,
                                  line_names: list):
        """
        Set digital lines by name

        # todo: better to take dictionary as argument? {line: value}
        :param array:
        :param line_names:
        :return:
        """
        if self.digital_line_names is None:
            raise ValueError("cannot set lines by name because self.digital_line_names is None")
        lines = [self.digital_line_names[n] for n in line_names]

        return self.set_digital_lines_by_index(array, lines)

    def set_analog_lines_by_address(self, array, addresses):
        """

        :param array:
        :param addresses:
        :return:
        """
        raise NotImplementedError("todo: this should be main function for setting analog lines once")

    def set_analog_once(self,
                        array: np.ndarray,
                        lines: Optional[list] = None,
                        lower_lims_volts: float = -5.0,
                        upper_lims_volts: float = 5.0):
        """
        Set analog lines once

        :param array:
        :param lines:
        :param lower_lims_volts: floating point number or array with same size as array
        :param upper_lims_volts:
        :return:
        """

        # check lines
        if lines is None:
            lines = list(range(self.n_analog_lines))

        # check array
        array = np.array(array)
        if array.shape != (len(lines),):
            raise ValueError(f"array must have shape {len(lines):d}, but had shape {array.shape}")

        # get lower voltage limits for all lines
        lower_lims_volts = np.atleast_1d(lower_lims_volts)
        if lower_lims_volts.size == 1:
            lower_lims_volts = np.tile(lower_lims_volts, len(lines))

        # get upper voltage limits for all lines
        upper_lims_volts = np.atleast_1d(upper_lims_volts)
        if upper_lims_volts.size == 1:
            upper_lims_volts = np.tile(upper_lims_volts, len(lines))

        # program DAQ
            for ii, l in enumerate(lines):
                try:  # ensure we dispose of task properly even if write fails because out of range
                    self._task_ao = daqmx.Task()
                    self._task_ao.CreateAOVoltageChan(self.analog_lines[l],
                                                      "",
                                                      lower_lims_volts[ii],
                                                      upper_lims_volts[ii],
                                                      daqmx.DAQmx_Val_Volts,
                                                      None)
                    self._task_ao.WriteAnalogScalarF64(True,
                                                       daqmx.DAQmx_Val_WaitInfinitely,
                                                       array[ii],
                                                       None)
                    self.last_known_analog_val[l] = array[ii]
                except daqmx.InvalidAODataWriteError as e:
                    print(e)
                finally:
                    self._task_ao.StopTask()
                    self._task_ao.ClearTask()

    def set_analog_lines_by_name(self,
                                 array: np.ndarray,
                                 line_names: list,
                                 lower_lims_volts: float = -5.0,
                                 upper_lims_volts: float = 5.0):
        """

        :param array:
        :param line_names:
        :param lower_lims_volts:
        :param upper_lims_volts:
        :return:
        """
        if isinstance(line_names, str):
            line_names = [line_names]

        if self.analog_line_names is None:
            raise ValueError("cannot set lines by name because self.analog_line_names is None")
        lines = [self.analog_line_names[n] for n in line_names]

        return self.set_analog_once(array, lines, lower_lims_volts, upper_lims_volts)

    def set_preset(self,
                   preset_name: str):
        """
        Set DAQ to preset value

        :param preset_name:
        :return:
        """
        if self.presets is None:
            raise ValueError("cannot set presets because self.presets is None")

        preset = self.presets[preset_name]

        # set digital lines, if there are any
        d = preset["digital"]
        if len(d) > 0:
            d_lines, d_arr = list(zip(*d.items()))
            self.set_digital_lines_by_name(np.array(d_arr), d_lines)

        # set analog lines, if there are any
        a = preset["analog"]
        if len(a) > 0:
            a_lines, a_arr = list(zip(*a.items()))
            self.set_analog_lines_by_name(np.array(a_arr), a_lines)

    def set_sine_wave(self,
                      amps: np.ndarray,
                      offs: np.ndarray,
                      frq: float,
                      lines: Optional[list] = None,
                      nsamples: int = 100):
        """
        Generate sine waves on selected analog channels. After setting, start with start_sequence() and stop with
        stop_sequence()

        :param amps:
        :param offs:
        :param frq:
        :param lines:
        :param nsamples:
        :return:
        """

        if lines is None:
            lines = list(range(self.n_analog_lines))

        amps = np.atleast_1d(amps)
        offs = np.atleast_1d(offs)

        ts = np.linspace(0, 1, nsamples)
        analog_array = amps[None, :] * np.sin(2 * np.pi * ts[:, None]) + offs[None, :]

        self._task_ao = daqmx.Task()
        self._task_ao.CreateAOVoltageChan(", ".join([d for ii, d in enumerate(self.analog_lines) if ii in lines]),
                                         "",
                                         -5.0,
                                         5.0,
                                         daqmx.DAQmx_Val_Volts,
                                         None)

        self._task_ao.CfgSampClkTiming("",
                                      frq * len(ts),
                                      daqmx.DAQmx_Val_Rising,
                                      daqmx.DAQmx_Val_ContSamps,
                                      analog_array.shape[0])

        samples_per_ch_ct = ct.c_int32()
        self._task_ao.WriteAnalogF64(analog_array.shape[0],
                                    False,
                                    10.0,
                                    daqmx.DAQmx_Val_GroupByScanNumber,
                                    analog_array,
                                    ct.byref(samples_per_ch_ct),
                                    None)

    def set_sequence(self,
                     digital_array: np.ndarray,
                     analog_array: np.ndarray,
                     sample_rate_hz: float,
                     digital_clock_source: str = "OnBoardClock",
                     analog_clock_source: str = "OnBoardClock",
                     digital_input_source: Optional[str] = None,
                     di_export_line: Optional[str] = None,
                     continuous: bool = True,
                     nrepeats: Optional[int] = None,
                     analog_read: bool = True,
                     pause_trigger_line: str = "/Dev1/PFI12",
                     interval: float = 0.,
                     pause_every_n: int = 1):
        """
        Set a sequence of digital (and optionally also analog) commands to the DAQ.

        This sequence can be started and stopped using start_sequence() and stop_sequence()

        :param digital_array: array of size n x n_channels
        :param analog_array: array of size m x m_channels. m and n need not be equal
        :param sample_rate_hz: sample rate at which digital samples will be generated
        :param digital_clock_source: clock source used for the digital task
        :param analog_clock_source: clock source used for the analog task
        :param digital_input_source: optional external line to perform edge detection on. The edge detection
          signal will be routed to di_export_line
        :param di_export_line: The edge detection signal read from digital_input_source is routed to this line. This
          line can then be used as a reference trigger for another line. Typically, this is used to trigger
          the analog program. In that cause analog_clock_source should be set to the same value as di_export_line
        :param continuous: if True, then sequence will be run repeatedly. If False, sequence will be run once.
        :param nrepeats: useful for specifying how much data to acquire with analog read
        :param analog_read: read all analog in ports during sequence
        :param pause_trigger_line: line which is used to pause the sequence when running with an interval of time
          between repeats
        :param interval: interval of time between repeats
        :param pause_every_n: perform n-repeats, then wait interval, then perform n more repeats
        :return:

        examples
        ####################
        # digital + analog sequences with same number of steps
        >>> daq = nidaq()
        >>> digital_array = np.zeros((10, 16)) # create digital array
        >>> digital_array[::2, 12] = 1 # alternative 0 and 1 on line 12
        >>> analog_array = np.zeros((10, 4)) # create analog array
        >>> analog_array[::2] = 3.0 # alternate all lines between 0 and 3 volts
        >>> dt = 100e-6 # time step in seconds
        >>> daq.set_sequence(digital_array, analog_array, 1/dt)
        >>> daq.start_sequence()
        >>> time.sleep(1)
        >>> daq.stop_sequence()

        # digital + analog sequence and using one of the digital lines to advance the analog sequence
        >>> daq = nidaq()
        >>> digital_array = np.zeros((10, 16)) # create digital array
        >>> digital_array[::2, 12] = 1 # alternative 0 and 1 on line 12. We will use this as the clock source for the analog signal
        >>> analog_array = np.zeros((5, 4)) # create analog array
        >>> analog_array[::2] = 3.0 # alternate all lines between 0 and 3 volts
        >>> dt = 100e-6 # time step in seconds
        >>> # assume that we hook up digital line 12 to the PFI1 input port
        >>> # We will export this signal to the PFI2 input port and use this as the clock source for the analog lines
        >>> daq.set_sequence(digital_array, analog_array, 1/dt, analog_clock_source="/Dev1/PFI2", digital_input_source="/Dev1/PFI1", di_export_line="/Dev1/PFI2")
        >>> daq.start_sequence()
        >>> time.sleep(1)
        >>> daq.stop_sequence()
        """

        if nrepeats is None:
            nrepeats = 1

        start_trigger = f"/{self.dev_name}/do/StartTrigger"
        counter = f"/{self.dev_name}/ctr0"

        # ######################
        # digital task
        # ######################
        self._task_do = daqmx.Task()
        self._task_do.CreateDOChan(self.digital_lines,
                                   "",
                                   daqmx.DAQmx_Val_ChanForAllLines)

        # clock source
        if continuous:
            repeat = daqmx.DAQmx_Val_ContSamps
        else:
            repeat = daqmx.DAQmx_Val_FiniteSamps

        self._task_do.CfgSampClkTiming(digital_clock_source,
                                       sample_rate_hz,
                                       daqmx.DAQmx_Val_Rising,
                                       repeat,
                                       digital_array.shape[0])

        # configure trigger source
        # self._task_do.CfgDigEdgeStartTrig(start_trigger, daqmx.DAQmx_Val_Rising)

        # Write the output waveform
        samples_per_ch_ct_digital = ct.c_int32()
        self._task_do.WriteDigitalLines(digital_array.shape[0],
                                        False,
                                        10.0,
                                        daqmx.DAQmx_Val_GroupByChannel,
                                        digital_array,
                                        ct.byref(samples_per_ch_ct_digital),
                                        None)

        # ######################
        # create pause trigger
        # when running a continuous sequence, this will be used to freeze sequence for an interval after each iteration
        # this supports sequences with high time resolution AND long time intervals between iterations
        # ######################
        if interval == 0:
            # ensure no pause trigger
            self._task_do.ResetPauseTrigType()
            interval_real = None
            self._task_ct = None
        else:
            # set pause trigger for digital line
            self._task_do.SetPauseTrigType(daqmx.Val_DigLvl)
            self._task_do.SetDigLvlPauseTrigSrc(pause_trigger_line)
            self._task_do.SetDigLvlPauseTrigWhen(daqmx.Val_High)

            # math for interval
            dt = 1 / sample_rate_hz
            N = int(interval // dt)
            interval_real = N * dt
            n = digital_array.shape[0]

            # pause trigger goes high for second half of last time-step in sequence
            delay = (n * pause_every_n - 0.5) * dt
            # pause triggers stays high until the second half of the last time-step before the next cycle
            duty_cycle = (N - n * pause_every_n) / N

            if duty_cycle <= 0:
                raise ValueError(f"Requested interval of {N:d} time steps is shorter"
                                 f" than sequence length = {n * pause_every_n:d} steps."
                                 " You must select a longer interval.")

            # configure counter to be used as pause trigger
            self._task_ct = daqmx.Task()

            self._task_ct.CreateCOPulseChanFreq(counter,
                                                "",
                                                daqmx.Val_Hz,
                                                daqmx.Val_Low,
                                                delay,
                                                1 / interval_real,
                                                duty_cycle)
            self._task_ct.CfgImplicitTiming(daqmx.DAQmx_Val_ContSamps, 0)
            self._task_ct.CfgDigEdgeStartTrig(start_trigger, daqmx.DAQmx_Val_Rising)
            self._task_ct.SetCOPulseTerm("", pause_trigger_line)

        # ######################
        # set up analog trigger line
        # if analog trigger source is not internal, then need to set up a digital input task that perform edge detection
        # on the digital_input_source port. Then, the edge detections must be routed to the analog trigger source port
        #
        # currently I manually route one of the digital out lines to the digital_input_source using a wire
        # ######################
        if digital_input_source is not None:
            self._task_di = daqmx.Task()
            self._task_di.CreateDIChan(digital_input_source,
                                       "",
                                       daqmx.DAQmx_Val_ChanForAllLines)

            # PFI lines are unbuffered, so must specify this
            self._task_di.CfgInputBuffer(0)

            # event on rising edge
            self._task_di.CfgChangeDetectionTiming(digital_input_source,
                                                   None,
                                                   daqmx.DAQmx_Val_ContSamps,
                                                   0)

            # export signal to another pin, so can use it to trigger or etc.
            if di_export_line is not None:
                self._task_di.ExportSignal(daqmx.DAQmx_Val_ChangeDetectionEvent, di_export_line)

            # configure start trigger
            self._task_di.CfgDigEdgeStartTrig(start_trigger, daqmx.DAQmx_Val_Rising)

        # ######################
        # analog ouput task
        # ######################
        if analog_array is None:
            self._task_ao = None
        else:
            # set tasks
            self._task_ao = daqmx.Task()
            self._task_ao.CreateAOVoltageChan(", ".join(self.analog_lines),
                                              "",
                                              -5.0,
                                              5.0,
                                              daqmx.DAQmx_Val_Volts,
                                              None)

            self._task_ao.CfgSampClkTiming(analog_clock_source,
                                           sample_rate_hz,
                                           daqmx.DAQmx_Val_Rising,
                                           repeat,
                                           analog_array.shape[0])

            self._task_ao.CfgDigEdgeStartTrig(start_trigger,
                                              daqmx.DAQmx_Val_Rising)

            # if analog task has only one step, then we need to add a second step,
            # otherwise WriteAnalogF64 will complain
            # This can happen if we are using a digital line to trigger the analog lines that only rarely change,
            if analog_array.shape[0] == 1:
                analog_array = np.concatenate((analog_array, analog_array), axis=0)

            samples_per_ch_ct = ct.c_int32()
            self._task_ao.WriteAnalogF64(analog_array.shape[0],
                                         False,
                                         10.0,
                                         daqmx.DAQmx_Val_GroupByScanNumber,
                                         analog_array,
                                         ct.byref(samples_per_ch_ct),
                                         None)

        # ######################
        # analog input task
        # ######################
        if not analog_read:
            self._task_ai = None
        else:
            if self._task_ai is not None:
                try:
                    self._task_ai.ClearTask()
                except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
                    pass

            self._task_ai = daqmx.Task()
            self._task_ai.CreateAIVoltageChan(", ".join(self.analog_input_line_names),
                                              "",
                                              daqmx.DAQmx_Val_Diff,
                                              -10.,
                                              10.,
                                              daqmx.DAQmx_Val_Volts,
                                              None
                                              )

            self._task_ai.CfgSampClkTiming(digital_clock_source,
                                           sample_rate_hz,
                                           daqmx.DAQmx_Val_Rising,
                                           daqmx.DAQmx_Val_FiniteSamps,
                                           digital_array.shape[0] * nrepeats
                                           )

            self._task_ai.CfgDigEdgeStartTrig(start_trigger,
                                              daqmx.DAQmx_Val_Rising)

        # todo: give option to block/wait for sequence to finish
        return interval_real

    def start_sequence(self):
        """
        Start a sequence prepared using set_sequence()

        :return:
        """
        if self._task_ct is not None:
            try:
                self._task_ct.StartTask()
            except daqmx.DAQmxFunctions.InvalidTaskError:
                pass

        if self._task_di is not None:
            try:
                self._task_di.StartTask()
            except daqmx.DAQmxFunctions.InvalidTaskError:
                pass

        if self._task_ao is not None:
            try:
                self._task_ao.StartTask()
            except daqmx.DAQmxFunctions.InvalidTaskError:
                pass

        if self._task_ai is not None:
            try:
                self._task_ai.StartTask()
            except daqmx.DAQmxFunctions.InvalidTaskError:
                pass

        if self._task_do is not None:
            try:
                self._task_do.StartTask()
            except daqmx.DAQmxFunctions.InvalidTaskError:
                pass

    def stop_sequence(self):
        """
        Stop a sequence

        :return:
        """

        # stop digital output task
        try:
            self._task_do.StopTask()
            self._task_do.ClearTask()
        except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
            pass

        # stop analog output task
        try:
            self._task_ao.StopTask()
            self._task_ao.ClearTask()
        except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
            pass

        # stop digital input task
        if self._task_di is not None:
            try:
                self._task_di.StopTask()
                self._task_di.ClearTask()
            except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
                pass

        # stop counter
        if self._task_ct is not None:
            try:
                self._task_ct.StopTask()
                self._task_ct.ClearTask()
            except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
                pass

        # don't stop analog task, otherwise problems
        # if self._task_ai is not None:
        #     try:
        #         self._task_ai.StopTask()
        #         # self._task_ai.ClearTask()
        #     except (daqmx.DAQmxFunctions.InvalidTaskError, AttributeError):
        #         pass

    def read_ai(self, n_samples, timeout=1., stop=True):
        """
        read analog input data recorded during sequence

        :param n_samples:
        :param timeout:
        :param stop:
        :return:
        """
        n_channels = len(self.analog_input_line_names)

        if n_channels != 1:
            raise NotImplementedError()

        data = np.zeros((n_samples, n_channels), dtype=np.float64)
        arr_size = data.size
        read = ct.c_int32()
        self._task_ai.ReadAnalogF64(n_samples,  # numSampsPerChan
                                    timeout,  # timeout in seconds
                                    daqmx.DAQmx_Val_GroupByChannel,  # fillMode
                                    data,  # readArray[]
                                    arr_size,  # arraySizeInSamps
                                    ct.byref(read),  # sampsPerchanRead
                                    None  # reserved
                                    )

        if stop:
            self._task_ai.StopTask()

        return data

# ###########################
# helper functions for working with line mappings and program arrays
# ###########################


def plot_daq_program(arr: np.ndarray,
                     line_map: Optional[dict] = None,
                     title: str = "",
                     **kwargs) -> Figure:
    """
    Plot DAQ program as an array

    :param arr: ntimes x nchannels array
    :param line_map: dictionary of line names
    :param title:
    :param kwargs:
    :return figh:
    """

    if line_map is None:
        line_names = {}
    k, v = zip(*list(line_map.items()))
    v = np.array(v)

    ticks = []
    for ii in range(arr.shape[1]):
        ind = np.argwhere(v == ii)
        if ind.size > 1:
            raise ValueError()
        elif ind.size == 1:
            ticks.append(Text(float(ii), 0, k[ind[0][0]]))
        else:
            ticks.append(Text(float(ii), 0, ""))

    figh = plt.figure(**kwargs)
    ax = figh.add_subplot(1, 1, 1)
    ax.set_title(title)

    ax.imshow(arr, aspect="auto", interpolation="none")

    ax.set_xticks(range(arr.shape[1]))
    ax.set_xticklabels(ticks)
    ax.set_xlabel("channel")
    ax.set_ylabel("time step")

    return figh


def get_line_names(map: dict) -> list:
    """
    Given a dictionary which specifies the mapping between line names and line indices, get list of line names such
    that line ii is called name[ii]

    :param map: a dictionary describing a mapping between line names and line indices
    :return line_names:
    """
    # get line names
    k = list(map.keys())
    v = np.array(list(map.values()))
    ind_max = np.max(v)

    nchannels = ind_max + 1

    line_names = []
    for ii in range(nchannels):

        inds = np.nonzero(v == ii)

        if len(inds) > 1:
            raise ValueError()

        ind = inds[0]
        if len(inds) == 1:
            ind = int(ind)
            name = k[ind]
        else:
            name = ""
        line_names.append(name)

    return line_names


def preset_to_array(preset: dict,
                    do_map: dict,
                    ao_map: dict,
                    n_digital_channels: Optional[int] = None,
                    n_analog_channels: Optional[int] = None):
    """
    Get arrays to program daq from presets

    :param preset: a dictionary with two keys: "digital" and "analog". preset["digital"] is another dictionary
      where the keys are some subset of the keys defined in do_map, and the values are the digital and analog voltages
      for the preset
    :param do_map: digital output map dictionary, where do_map["line_name"] = line index
    :param ao_map:
    :param n_digital_channels: size used to generate array. If not specified use the largest value in do_map
    :param n_analog_channels: size used to generate array. If not specific use the largest value in ao_map
    :return digital_array, analog_array:
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

def format_config_data(digital_map: dict,
                       analog_map: dict,
                       presets: dict):
    """
    Prepare data to be saved in configuration file. This helper function is useful if you want to save this data
    as one entry in a larger configuration file

    :param digital_map:
    :param analog_map:
    :param presets:
    :return:
    """

    now = datetime.datetime.now()
    tstamp = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d};{now.minute:02d};{now.second:02d}"
    data = {"timestamp": tstamp,
            "analog_map": analog_map,
            "digital_map": digital_map,
            "presets": presets}
    return data


def save_config_file(fname: str,
                     digital_map: dict,
                     analog_map: dict,
                     presets: dict):
    """
    Save configuration data to json file

    :param fname:
    :param analog_map:
    :param digital_map:
    :param presets:
    :return:
    """

    data = format_config_data(digital_map, analog_map, presets)
    with open(fname, "w") as f:
        json.dump(data, f, indent="\t")


def load_config_file(fname: str,
                     component: Optional[str] = None) -> (dict, dict, dict, str):
    """
    load configuration data from json file

    :param fname:
    :param component:
    :return analog_map, digital_map, presets, tstamp:
    """
    with open(fname, "r") as f:
        data = json.load(f)

    if component is not None:
        data = data[component]

    tstamp = data["timestamp"]
    analog_map = data["analog_map"]
    digital_map = data["digital_map"]
    presets = data["presets"]

    return digital_map, analog_map, presets, tstamp


if __name__ == "__main__":
    # example configuration

    # DAQ line mapping
    daq_do_map = {"odt_cam": 0,
                  "odt_shutter": 9,
                  "odt_laser": 2,
                  "sim_cam": 5,
                  "sim_shutter": 6,
                  "blue_laser": 8,
                  "green_laser": 7,
                  "red_laser": 1,
                  "dmd_enable": 4,
                  "dmd_advance": 3,
                  "odt_cam_enable": 10,
                  "signal_monitor": 11,
                  "analog_trigger": 12
                  }

    # analog lines
    daq_ao_map = {"vc_mirror_x": 0,
                  "vc_mirror_y": 1,
                  "z_stage": 2,
                  "z_stage_monitor": 3}

    # dictionary of presets
    # each preset is a dictionary with keys "digital" and "analog"
    # preset["blue"]["digital"] is a dictionary where the keys are the channel names
    presets = {"off": {"digital": {"odt_cam": 0,
                                   "odt_shutter": 0,
                                   "odt_laser": 0,
                                   "sim_cam": 0,
                                   "sim_shutter": 0,
                                   "blue_laser": 0,
                                   "green_laser": 0,
                                   "red_laser": 0,
                                   "dmd_enable": 0,
                                   "dmd_advance": 0},
                       "analog": {}
                       },
               "blue": {"digital": {"odt_shutter": 0,
                                    "odt_laser": 0,
                                    "sim_shutter": 1,
                                    "blue_laser": 1,
                                    "green_laser": 0,
                                    "red_laser": 0},
                        "analog": {"vc_mirror_x": -0.25,
                                   "vc_mirror_y": 0}},
               "green": {"digital": {"odt_shutter": 0,
                                     "odt_laser": 0,
                                     "sim_shutter": 1,
                                     "blue_laser": 0,
                                     "green_laser": 1,
                                     "red_laser": 0},
                         "analog": {"vc_mirror_x": -4.75,
                                    "vc_mirror_y": -1.05}},
               "red": {"digital": {"odt_cam": 0,
                                   "odt_shutter": 0,
                                   "sim_shutter": 1,
                                   "blue_laser": 0,
                                   "green_laser": 0,
                                   "red_laser": 1},
                       "analog": {"vc_mirror_x": -0.7,
                                  "vc_mirror_y": 0}},
               "odt": {"digital": {"odt_shutter": 1,
                                   "odt_laser": 1,
                                   "sim_shutter": 0,
                                   "blue_laser": 0,
                                   "green_laser": 0,
                                   "red_laser": 0},
                       "analog": {}}
               }

    save_config_file("daq_config_example.json", daq_do_map, daq_ao_map, presets)
