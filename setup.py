from setuptools import setup, find_packages
import sys
from os import path

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md")) as f:
    long_description = f.read()

required_pkgs = ['numpy',
                 'scipy',
                 'matplotlib',
                 'pandas',
                 'scikit-image',
                 'joblib',
                 'psutil',
                 'tifffile',
                 'zarr',
                 'h5py',
                 'dask',
                 'localize_psf @ git+https://git@github.com/qi2lab/localize-psf@master#egg=localize_psf'
                 ]

# extras
# to install cucim manually pip install "git+https://github.com/rapidsai/cucim.git@v22.12.00#egg=cucim&subdirectory=python/cucim"
extras = {'expt_ctrl': ['PyDAQmx'],
          'gpu': ['cupy-cuda11x', # assuming CUDA 11.2 or later. Otherwise, manually install
                  'cucim @ git+https://github.com/rapidsai/cucim.git@v22.04.00#egg=cucim&subdirectory=python/cucim'
                  ]
          }

# check what platform we are on
if sys.platform == 'win32':
    extras['expt_ctrl'].append(['pywinusb>=0.4.2'])

setup(
    name='mcsim',
    version='1.3.0',
    description="A package for simulating and controlling a multicolor structured illumination"
                " microscopy (SIM) experiment using a DLP6500 digital micromirror device (DMD)"
                " and performing SIM reconstruction.",
    long_description=long_description,
    author='Peter T. Brown, qi2lab',
    author_email='ptbrown1729@gmail.com',
    packages=find_packages(include=['mcsim', 'mcsim.*']),
    python_requires='>=3.9',
    install_requires=required_pkgs,
    extras_require=extras)
