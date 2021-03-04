"""
Set up Optotune MRE2 mirror using their python SDK.
"""
import optoMDC

mre2 = optoMDC.connect()
mre2.reset()

# set up x-channel
ch0 = mre2.Mirror.Channel_0

ch0.InputConditioning.SetGain(0.04)
# ch0.InputConditioning.GetGain()
ch0.InputConditioning.SetOffset(0.)
# ch0.InputConditioning.GetOffset()
ch0.Analog.SetAsInput()
ch0.SetControlMode(optoMDC.Units.XY) # XY = "Closed Loop", Current = "Open Loop"
# ch0.GetControlMode()
# ch0.Manager.CheckSignalFlow()

# set up y-channel
ch1 = mre2.Mirror.Channel_1
ch1.InputConditioning.SetGain(0.01)
ch1.InputConditioning.SetOffset(0.)
ch1.Analog.SetAsInput()
ch1.SetControlMode(optoMDC.Units.XY) # XY = "Closed Loop", Current = "Open Loop"
#ch1.Manager.CheckSignalFlow()

print("Initialized Optotune Mirror")