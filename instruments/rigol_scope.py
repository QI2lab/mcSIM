'''
Extract data from Rigol DS1054Z scope

install libusb driver using Zadig
pip install libusb
pip install pyusb
pip install python-usbtmc

pip install pyvisa
pip install pyvisa-py

Code inspired by https://www.cibomahto.com/2010/04/controlling-a-rigol-oscilloscope-using-linux-and-python/
'''


""" Example program to plot the Y-T data from Channel 1"""

import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import pickle
# import usb
# import usbtmc
import visa
import analysis_tools as tools

def acquire_trace(dev, channel, mode='RAW'):
    """
    Acquire trace from Rigol DS1054Z oscilloscope

    :param dev:
    :param channel:
    :param mode: 'NORM', 'RAW', or 'MAX'. 'MAX' will give NORM or RAW, depending on if scope is running or stopped
    :return:
    """

    # can only read raw, i.e. data from memory, when stopped
    if mode == 'RAW':

        dev.write(":STOP")

    # set up read
    dev.write(":WAV:SOUR CHAN%d" % channel)
    # dev.write(":WAV:MODE NORM")
    dev.write(":WAV:MODE %s" % mode)
    dev.write(":WAV:FORM BYTE")  # waveform occupies one byte
    # dev.write(":WAV:FORM WORD") # waveform occupies two bytes, only lower 8 bits valid

    # how many samples do we need to read?
    # todo: may need to modify if in NORM instead of RAW mode
    nsamples = int(dev.query("ACQ:MDEP?"))
    # todo: can I get the max allowed value by querying scope?
    n_per_read = 250000
    niters = int(np.ceil(nsamples / n_per_read))

    print("starting read, %d iterations" % niters)
    tstart = time.process_time()

    # to store results
    vdat_adc = []
    n_end = 0
    while n_end < nsamples:
        n_start = n_end + 1
        n_end = np.min([n_start + n_per_read - 1, nsamples])

        dev.write(":WAV:START %d" % n_start)
        dev.write(":WAV:STOP %d" % n_end)

        dev.write(":WAV:DATA?")
        # if use ascii then will throw errors if above 128
        raw_dat = dev.read(encoding='latin1')

        denoter = raw_dat[:11]
        nbytes = int(denoter[2:])

        # last point is terminator (0X0A)
        vdat_adc = vdat_adc + list(map(ord, raw_dat[11:-1]))
    tend = time.process_time()
    print("finished reading data, %0.3fs total, %0.3fs per read" % (tend - tstart, (tend - tstart) / niters))


    # vscale / 25 = vrange / 200
    yinc = float(dev.query("WAV:YINC?"))
    yorigin = float(dev.query("WAV:YOR?"))
    yref = int(dev.query(":WAV:YREF?"))
    # xinc = tscale / 100 in Normal mode, or 1/sample_rate in raw mode
    xinc = float(dev.query(":WAV:XINC?"))
    xorigin = float(dev.query(":WAV:XOR?"))
    xref = float(dev.query(":WAV:XREF?"))

    # format of output is "#NXXXXXXXXX"
    # where XXXXXXXXX is the number of bytes included

    # programming manual provided method of converting to voltage. See page 2-215 in section on :WAVeform:DATA? command
    vdat = (np.asarray(vdat_adc) - yorigin - yref) * yinc
    times = xorigin + np.arange(vdat.size) * xinc

    # alternatively, can read in ASCII values, check vdat conversion
    # dev.write(":WAV:FORM ASC")
    # raw_dat = dev.query(":WAV:DATA?")
    # denoter = raw_dat[:11]
    # nbytes = int(denoter[2:])
    # dat = np.asarray(list(map(float, raw_dat[11:].split(','))))

    return vdat, times

# # first find devices
# devs = usb.core.find()
# # get vendor and product ids
# PID = devs.idProduct
# VID = devs.idVendor

# PID = 1230
# VID = 6833
# dev = usbtmc.Instrument(VID, PID)
# works, but throws timeout error whenever I try to dev.ask() or dev.write()


# resources = visa.ResourceManager('@py')
# dev_names = resources.list_resources()
# dev = resources.open_resource(dev_names[0])
#     Traceback (most recent call last):
#   File "C:\Program Files\Python\Python37\lib\site-packages\IPython\core\interactiveshell.py", line 3319, in run_code
#     exec(code_obj, self.user_global_ns, self.user_ns)
#   File "<ipython-input-15-74033834a8ea>", line 1, in <module>
#     scope = rm.open_resource(rm.list_resources()[0])
#   File "C:\Program Files\Python\Python37\lib\site-packages\pyvisa\highlevel.py", line 1771, in open_resource
#     res.open(access_mode, open_timeout)
#   File "C:\Program Files\Python\Python37\lib\site-packages\pyvisa\resources\resource.py", line 218, in open
#     self.session, status = self._resource_manager.open_bare_resource(self._resource_name, access_mode, open_timeout)
#   File "C:\Program Files\Python\Python37\lib\site-packages\pyvisa\highlevel.py", line 1725, in open_bare_resource
#     return self.visalib.open(self.session, resource_name, access_mode, open_timeout)
#   File "C:\Program Files\Python\Python37\lib\site-packages\pyvisa-py\highlevel.py", line 194, in open
#     sess = cls(session, resource_name, parsed, open_timeout)
#   File "C:\Program Files\Python\Python37\lib\site-packages\pyvisa-py\sessions.py", line 213, in __init__
#     self.after_parsing()
#   File "C:\Program Files\Python\Python37\lib\site-packages\pyvisa-py\usb.py", line 201, in after_parsing
#     self.parsed.serial_number)
#   File "C:\Program Files\Python\Python37\lib\site-packages\pyvisa-py\protocols\usbtmc.py", line 261, in __init__
#     self.usb_dev.set_configuration()
#   File "C:\Program Files\Python\Python37\lib\site-packages\usb\core.py", line 869, in set_configuration
#     self._ctx.managed_set_configuration(self, configuration)
#   File "C:\Program Files\Python\Python37\lib\site-packages\usb\core.py", line 102, in wrapper
#     return f(self, *args, **kwargs)
#   File "C:\Program Files\Python\Python37\lib\site-packages\usb\core.py", line 148, in managed_set_configuration
#     self.backend.set_configuration(self.handle, cfg.bConfigurationValue)
#   File "C:\Program Files\Python\Python37\lib\site-packages\usb\backend\libusb0.py", line 493, in set_configuration
#     _check(_lib.usb_set_configuration(dev_handle, config_value))
#   File "C:\Program Files\Python\Python37\lib\site-packages\usb\backend\libusb0.py", line 431, in _check
#     raise USBError(errmsg, ret)
# usb.core.USBError: [Errno None] b'libusb0-dll:err [set_configuration] could not set config 1: win error: The parameter is incorrect.\r\n'

# number of measurements to take
nreps = 2

# Can't get USB to work, so let's try ethernet with IVSA
resources = visa.ResourceManager('@py')
#dev_names = resources.list_resources() # ethernet doesn't show up for some reason
# using browser, find device information
dev = resources.open_resource('TCPIP::169.254.56.27::INSTR')
# set timeout in ms
dev.timeout = 60000
id = dev.query("*IDN?")

# acquisition settings
#dev.write(":ACQ:MDEP AUTO")
dev.write(":RUN")
dev.write(":ACQ:TYPE NORM")
dev.write(":ACQ:MDEP 1200000") # allowed values when only one channel is on 12000|120000|1200000|12000000|24000000

# set up measurement settings for channel
dev.write(":CHAN4:BWL OFF")
dev.write(":CHAN4:COUP DC")
dev.write(":CHAN4:DISP")
dev.write(":CHAN4:INV ON")
dev.write(":CHAN4:PROB 1")

# Get voltage and time scales. NOTE: these are useful, but not the full information. They are more relevant for
# Normal mode (i.e. when referring to the WAV system, where normal mode means capturing what is on the screen,
# vs. raw mode, which is relevant for reading internal memory)
# note that these return the size of the major divisions
# there are 8 major divisions for the voltage axis
# NOTE: these are not the relevant things in Raw mode
vscale = float(dev.query(":CHAN4:SCAL?"))
vrange = float(dev.query(":CHAN4:RANG?")) # 8* vscale
voffset = float(dev.query(":CHAN4:OFFS?"))

# samples/s
srate = float(dev.query(":ACQ:SRAT?"))
# there are 12 major divisions for the time axis
tscale = float(dev.query(":TIM:SCAL?"))
toffset = float(dev.query(":TIM:OFFS?"))
tcal = float(dev.query(":CHAN4:TCAL?"))

# take measurements
times_all = []
vdat_all = []
for ii in range(nreps):
    dev.write(":RUN")
    time.sleep(10)
    vdat, times = acquire_trace(dev, 4, mode='RAW')

    times_all.append(times)
    vdat_all.append(vdat)
dev.write(":RUN")

# ffts for all measurements
ffts = []
frqs = []
for t, v in zip(times_all, vdat_all):
    ffts.append(fftpack.fftshift(fftpack.fft(v)))
    dt = t[1] - t[0]
    frqs.append(tools.get_fft_frqs(len(v), dt))

# save results
now = datetime.datetime.now()
#fname = "%04d_%02d_%02d_%02d;%02d;%02d_rigol_ds1054z.pkl" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
fname = tools.get_unique_name('2020_02_06_spifi_ntimes=30_period=6_commensurate.pkl')
with open(fname, 'wb') as f:
    data = {'time': times_all[0], 'voltage': np.asarray(vdat_all), 'frq': frqs[0], 'voltage_ft': np.asarray(ffts)}
    pickle.dump(data, f)

# plot results
plt.figure()

plt.subplot(1, 2, 1)
for t, v in zip(times_all, vdat_all):
    plt.plot(t, v)

plt.xlabel('time (s)')
plt.ylabel('signal (v)')
plt.ylim([voffset - 0.5*vrange, voffset + 0.5*vrange])

plt.subplot(1, 2, 2)
for f, ft in zip(frqs, ffts):
    plt.semilogy(f, np.abs(ft))
plt.xlabel('frq (hz)')
plt.ylabel('log|fft|')
plt.xlim([-10e3, 10e3])

plt.suptitle('%.3g points at %.0f Ms/S, dt=%0.0fns' % (vdat.size, srate / 1e6, dt/1e-9))
plt.show()

