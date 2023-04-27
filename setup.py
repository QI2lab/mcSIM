from setuptools import setup, find_packages
import sys
from os import path
from mcsim import __version__

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
                 'dask-image',
                 'localize_psf @ git+https://git@github.com/qi2lab/localize-psf@master#egg=localize_psf'
                 ]

# extras
extras = {'expt_ctrl': ['PyDAQmx'],
          'gpu': [# assuming 11.2 <= CUDA version < 12. Otherwise, manually install with
                  # conda install -c conda-forge cupy cudatoolkit=11.8
                  'cupy-cuda11x',
                  # periodically update version of cucim checkout by hand. The symbol following @ should be
                  # the release tag name
                  # to install cucim manually
                  # pip install "git+https://github.com/rapidsai/cucim.git@v22.12.00#egg=cucim&subdirectory=python/cucim"
                  'cucim @ git+https://github.com/rapidsai/cucim.git@v23.02.00#egg=cucim&subdirectory=python/cucim'
                  # 'cucim @ git+https://github.com/rapidsai/cucim.git@v22.04.00#egg=cucim&subdirectory=python/cucim'
                  ]

          }

# check what platform we are on
if sys.platform == 'win32':
    extras['expt_ctrl'].append(['pywinusb>=0.4.2'])

setup(
    name='mcsim',
    version=__version__,
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
