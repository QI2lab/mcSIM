"""
Setup Optotune MRE2 mirror using their python SDK
Tested with version MR-E-2_PythonSDK_1.2.4065

After downloading the SDK from the Optotune website, navigate to the folder and install using
pip install optoKummenberg-0.18.3974-py3-none-any.whl
pip install optoMDC-1.2.4065-py3-none-any.whl
"""
import optoMDC


def initialize_mre2():
    mre2 = optoMDC.connect()
    mre2.reset()

    # set up x-channel
    ch0 = mre2.Mirror.Channel_0

    ch0.InputConditioning.SetGain(0.04)
    # ch0.InputConditioning.GetGain()
    ch0.InputConditioning.SetOffset(0.)
    # ch0.InputConditioning.GetOffset()
    ch0.Analog.SetAsInput()
    ch0.SetControlMode(optoMDC.Units.XY)  # XY = "Closed Loop", Current = "Open Loop"
    # ch0.GetControlMode()
    # ch0.Manager.CheckSignalFlow()

    # set up y-channel
    ch1 = mre2.Mirror.Channel_1
    ch1.InputConditioning.SetGain(0.01)
    ch1.InputConditioning.SetOffset(0.)
    ch1.Analog.SetAsInput()
    ch1.SetControlMode(optoMDC.Units.XY)  # XY = "Closed Loop", Current = "Open Loop"
    # ch1.Manager.CheckSignalFlow()


if __name__ == "__main__":
    initialize_mre2()
    print("initialized voice coil mirror")
