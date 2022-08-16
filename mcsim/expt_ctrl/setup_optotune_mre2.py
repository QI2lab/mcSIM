"""
Set up Optotune MRE2 mirror using their python SDK. This file is called by MicroManager using the beanshell script
MMStartup.bsh

Installing Optotune SDK (MR-E-2_PythonSDK_1.2.3933) to your python environment:
(1) The Optotune voice coil mirror SDK can be downloaded from their website https://www.optotune.com/downloads after
registering
(2) Since Optotune does not provide a setup.py file to pip install their package after downloading, we can make a
    simple one by hand in the top directory of their SDK.
    setup.py:
    ```
    from setuptools import setup, find_packages

    setup(
       name="optotune_sdk",
        packages=find_packages(where=".")
        )
    '''
(3) Add an empty file named "__init__.py" to the folder "inflect"
(4) navigate to the top directory and type "pip install ."
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

if __name__ == "__main__":
    initialize_mre2()
    print("initialized voice coil mirror")