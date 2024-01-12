"""
Tools for simulating diffraction from a digital micromirror device (DMD).
There are three important effects to consider:
(1) diffraction from the underlying DMD diffraction grating
(2) diffraction from whatever pattern of mirrors the DMD displays
(3) an efficiency envelope imposed by the diffraction from each mirror individually. This envelope is peaked
at the specular reflection condition for the mirror. When light diffracts in the same direction as the peak,
we say the blaze condition is satisfied.

The simulate_dmd_dft() function is the most useful function for computing all three effects. Given
geometry information (input direction, DMD pitch, diffraction order of interest, etc.) and a mirror pattern,
this provides the diffracted electric field at a number of angles, where the angles are related to the DFT
frequencies. In some sense, this provides the complete information about the diffraction pattern. Other angles
can be generated through exact sinc interpolation (i.e. DFT analog of the Shannon-Whittaker interpolation formula).
This interpolation can be performed using interpolate_dmd_data() for arbitrary angles. Doing this interpolation
is mostly useful for understanding Fourier broadening of diffraction peaks.

For direct simulation of arbitrary output angles, the simulate_dmd() function performs a brute force simulation
which is essentially an O(n^2) numerical discrete Fourier transform (DFT) plus the effect of the blaze envelope.
This is vastly less efficient than simulate_dmd_dft(), since the FFT algorithm is O(nlog(n)). It essentially
provides the same services as the combination of simulate_dmd_dft() and interpolate_dmd_data()

When designing a DMD system, the most important questions are how to set the input and output angles in such
a way that the blaze condition is satisfied. Many of the other tools provided here can be used to answer these
questions. For example, find_combined_condition() determines what pairs of input/output angles satisfy both
the blaze and diffraction condition. solve_1color_1d() is a wrapper which solves the same problem along the x-y
direction (i.e. for typical operation of the DMD).  get_diffracted_output_uvec() computes the angles of diffraction
orders for a given input direction. etc...

When simulating a periodic pattern such as used in Structured Illumination Microscopy (SIM), the tools found in
dmd_pattern.py may be more suitable.

Coordinate systems
####################
We adopt a coordinate system with x- and y- axes along the primary axes of the DMD chip (i.e. determined
by the periodic mirror array), and z- direction is positive pointing away from the DMD face. This way the unit
vectors describing the direction of an incoming plane waves has negative z-component, and the unit vector of
an outgoing plane wave has positive z-component. We typically suppose the mirrors swivel about the axis
n = [1, 1, 0]/sqrt(2), i.e. diagonal to the DMD axes, by angle +/- gamma. This ensures that light incident in
the x-y (x minus y) plane stays in plane after diffraction (for the blazed order)

In addition to the xyz coordinate system, we also use two other convenient coordinate systems.
1. the mpz coordinate system:
This coordinate system is convenient for dealing with diffraction from the DMD, as discussed above. Note
that the mirrors swivel about the ep direction
em = (ex - ey) / sqrt(2); ep = (ex + ey) / sqrt(2)
2. the 123 or "mirror" coordinate system:
This coordinate system is specialized to dealing with the blaze condition. Here the unit vector e3 is the normal to the
DMD mirror, e2 is along the (x+y)/sqrt(2) direction, and e1 is orthogonal to these two. Since e3 is normal
to the DMD mirrors this coordinate system depends on the mirror swivel angle.

In whichever coordinate system, if we want to specify directions we have the choice of working with either
unit vectors or an angular parameterization. Typically unit vectors are easier to work with, although angles
may be easier to interpret. We use different angular parameterizations for incoming and outgoing unit vectors.
For example, in the xy coordinate system we use
a = az * [tan(tx_a), tan(ty_a), -1]
b = |bz| * [tan(tb_x), tan(tb_y), 1]

If light is incident towards the DMD as a plane wave from some direction determined by a unit vector, a, then it
is then diffracted into different output directions depending on the spatial frequencies of the DMD pattern.
Call these directions b(f).

If the DMD is tilted, the DMD pattern frequencies f will not exactly match the optical system frequencies.
In particular, although the DMD pattern will have components at f and -f the optical system frequencies will
not be perfectly centered on the optical axis.
"""
from typing import Union, Optional, Sequence, Callable
from pathlib import Path
import numpy as np
import dask
from dask.diagnostics import ProgressBar
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle

_cupy_available = True
try:
    import cupy as cp
except ImportError:
    cp = np
    _cupy_available = False

array = Union[np.ndarray, cp.ndarray]

# ###########################################
# main simulation functions
# ###########################################
_dlp_1stgen_axis = (1/np.sqrt(2), 1/np.sqrt(2), 0.)


# todo: migrate functions to this class
class DMD:
    def __init__(self,
                 dx: float,
                 dy: float,
                 wx: float,
                 wy: float,
                 gamma_on: float,
                 gamma_off: float,
                 rot_axis_on: Sequence[float] = _dlp_1stgen_axis,
                 rot_axis_off: Sequence[float] = _dlp_1stgen_axis,
                 nx: Optional[int] = None,
                 ny: Optional[int] = None):
        """

        :param dx: spacing between DMD pixels in the x-direction. Same units as wavelength.
        :param dy: spacing between DMD pixels in the y-direction
        :param wx: width of mirrors in the x-direction. Must be <= dx.
        :param wy: width of mirrors in y-direction
        :param gamma_on: DMD mirror angle in radians
        :param gamma_off:
        :param rot_axis_on:
        :param rot_axis_off:
        """

        if dx < wx or dy < wy:
            raise ValueError('w must be <= d.')

        self.gamma_on = float(gamma_on)
        self.gamma_off = float(gamma_off)
        self.dx = float(dx)
        self.dy = float(dy)
        self.wx = float(wx)
        self.wy = float(wy)
        self.rot_axis_on = np.asarray(rot_axis_on)
        self.rot_axis_off = np.asarray(rot_axis_off)
        self.size = (ny, nx)

    def simulate_pattern(self,
                         wavelength: float,
                         pattern: array,
                         uvec_in: array,
                         uvecs_out: array,
                         zshifts: Optional[array] = None,
                         phase_errs: Optional[array] = None,
                         efield_profile: Optional[array] = None):
        """

        :param wavelength:
        :param pattern:
        :param uvec_in: (ax, ay, az) direction of plane wave input to DMD
        :param uvecs_out: N x 3 array. Output unit vectors where diffraction should be computed.
        :param zshifts: if DMD is assumed to be non-flat, give height profile here. Array of the same size as pattern
        :param phase_errs: direct phase errors per mirror. This is an alternative way to provide aberration information
        compared with zshifts
        :param efield_profile:
        :return:
        """
        pass

class DLP7000(DMD):
    def __init__(self):
        super(DLP7000, self).__init__(13.68,
                                      13.68,
                                      13.68,
                                      13.68,
                                      12 * np.pi / 180,
                                      -12 * np.pi / 180,
                                      rot_axis_on=_dlp_1stgen_axis,
                                      rot_axis_off=_dlp_1stgen_axis,
                                      nx=1024,
                                      ny=768)

class DLP6500(DMD):
    def __init__(self):
        super(DLP6500, self).__init__(7.56,
                                      7.56,
                                      7.56,
                                      7.56,
                                      12 * np.pi/180,
                                      -12 * np.pi/180,
                                      rot_axis_on=_dlp_1stgen_axis,
                                      rot_axis_off=_dlp_1stgen_axis,
                                      nx=1920,
                                      ny=1080)

class DLP5500(DMD):
    def __init__(self):
        super(DLP5500, self).__init__(10.8,
                                      10.8,
                                      10.8,
                                      10.8,
                                      12 * np.pi / 180,
                                      -12 * np.pi / 180,
                                      rot_axis_on=_dlp_1stgen_axis,
                                      rot_axis_off=_dlp_1stgen_axis,
                                      nx=1024,
                                      ny=768)
class DLP4710(DMD):
    def __init__(self):
        r1 = get_rot_mat(np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0]), 12 * np.pi / 180)
        r2 = get_rot_mat(np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0]), 12 * np.pi / 180)

        # convert on rotation matrix to equivalent rotation about single axis
        r_on = r2.transpose().dot(r1)
        rot_axis_on, gamma_on = get_rot_mat_angle_axis(r_on)
        r_off = r2.dot(r1)
        rot_axis_off, gamma_off = get_rot_mat_angle_axis(r_off)

        super(DLP4710, self).__init__(5.4,
                                      5.4,
                                      5.4,
                                      5.4,
                                      gamma_on,
                                      gamma_off,
                                      rot_axis_on=rot_axis_on,
                                      rot_axis_off=rot_axis_off,
                                      nx=1920,
                                      ny=1080)


def simulate_dmd(pattern: array,
                 wavelength: float,
                 gamma_on: float,
                 gamma_off: float,
                 dx: float,
                 dy: float,
                 wx: float,
                 wy: float,
                 uvec_in: array,
                 uvecs_out: array,
                 zshifts: Optional[array] = None,
                 phase_errs: Optional[array] = None,
                 efield_profile: Optional[array] = None,
                 rot_axis_on: Sequence[float] = _dlp_1stgen_axis,
                 rot_axis_off: Sequence[float] = _dlp_1stgen_axis) -> (array, array, array):
    """
    Simulate plane wave diffracted from a digital mirror device (DMD) naively. In most cases this function is not
    the most efficient to use! When working with SIM patterns it is much more efficient to rely on the tools
    found in dmd_patterns

    We assume that the body of the device is in the xy plane with the negative z-unit vector defining the plane's
    normal. This means incident unit vectors have positive z-component, and outgoing unit vectors have negative
    z-component. We suppose the device has rectangular pixels with sides parallel to the x- and y-axes.
    We further suppose a given pixel (centered at (0,0)) swivels about the vector n = [1, 1, 0]/sqrt(2)
    by angle gamma, i.e. the direction x-y is the most interesting one.
 
    :param pattern: an NxM array. Dimensions of the DMD are determined from this. As usual, the upper left
     hand corner if this array represents the smallest x- and y- values
    :param wavelength: choose any units as long as consistent with dx, dy, wx, and wy.
    :param gamma_on: DMD mirror angle in radians
    :param gamma_off:
    :param dx: spacing between DMD pixels in the x-direction. Same units as wavelength.
    :param dy: spacing between DMD pixels in the y-direction. Same units as wavelength.
    :param wx: width of mirrors in the x-direction. Must be <= dx.
    :param wy: width of mirrors in the y-direction. Must be <= dy.
    :param uvec_in: (ax, ay, az) direction of plane wave input to DMD
    :param uvecs_out: N x 3 array. Output unit vectors where diffraction should be computed.
    :param zshifts: if DMD is assumed to be non-flat, give height profile here. Array of the same size as pattern
    :param phase_errs: direct phase errors per mirror. This is an alternative way to provide aberration information
      compared with zshifts
    :param efield_profile: electric field values (amplitude and phase) across the DMD
    :param rot_axis_on:
    :param rot_axis_off:
    :return efields, sinc_efield_on, sinc_efield_off, diffraction_efield:
    """

    if isinstance(pattern, cp.ndarray):
        xp = cp
    else:
        xp = np

    # check input arguments are sensible
    if not pattern.dtype == bool:
        raise TypeError('pattern must be of type bool.')

    if dx < wx or dy < wy:
        raise ValueError('w must be <= d.')

    if zshifts is None:
        zshifts = xp.zeros(pattern.shape)
    zshifts = xp.array(zshifts)

    if phase_errs is None:
        phase_errs = xp.zeros(pattern.shape)
    phase_errs = xp.asarray(phase_errs)

    if efield_profile is None:
        efield_profile = xp.ones(pattern.shape)
    efield_profile = xp.array(efield_profile)

    uvec_in = xp.array(uvec_in)
    uvecs_out = xp.atleast_2d(uvecs_out)

    # DMD pixel coordinates using origin at center FFT convention
    ny, nx = pattern.shape
    mxmx, mymy = xp.meshgrid(xp.arange(nx) - nx // 2,
                             xp.arange(ny) - ny // 2)

    # function to do computation for each output unit vector
    def calc_output_angle(bvec):
        # incoming minus outgoing unit vectors
        bma = bvec - uvec_in.squeeze()

        # efield phase for each DMD pixel
        efield_per_mirror = efield_profile * \
                            xp.exp(-1j * 2*np.pi / wavelength * (dx * mxmx * bma[0] +
                                                                 dy * mymy * bma[1] +
                                                                 zshifts * bma[2]) +
                                   1j * phase_errs)

        # get envelope functions for "on" and "off" states
        sinc_efield_on = wx * wy * blaze_envelope(wavelength, gamma_on, wx, wy, bma, rot_axis_on)
        sinc_efield_off = wx * wy * blaze_envelope(wavelength, gamma_off, wx, wy, bma, rot_axis_off)

        # final summation
        efs = xp.sum(efield_per_mirror * (sinc_efield_on * pattern + sinc_efield_off * (1 - pattern)))

        return efs, sinc_efield_on, sinc_efield_off

    # get shape want output arrays to be
    output_shape = uvecs_out.shape[:-1]
    # reshape bvecs to iterate over
    bvecs_to_iterate = xp.reshape(uvecs_out, (np.prod(output_shape), 3))

    # simulate
    with ProgressBar():
        r = [dask.delayed(calc_output_angle)(bvec) for bvec in bvecs_to_iterate]
        results = dask.compute(*r)

    # unpack results for all output directions
    efields, sinc_efield_on, sinc_efield_off = zip(*results)
    efields = xp.asarray(efields).reshape(output_shape)
    sinc_efield_on = xp.asarray(sinc_efield_on).reshape(output_shape)
    sinc_efield_off = xp.asarray(sinc_efield_off).reshape(output_shape)

    return efields, sinc_efield_on, sinc_efield_off


def simulate_dmd_dft(pattern: array,
                     efield_profile: array,
                     wavelength: float,
                     gamma_on: float,
                     gamma_off: float,
                     dx: float,
                     dy: float,
                     wx: float,
                     wy: float,
                     uvec_in,
                     order: Sequence[int],
                     dn_orders: int = 0,
                     rot_axis_on: Sequence[float] = _dlp_1stgen_axis,
                     rot_axis_off: Sequence[float] = _dlp_1stgen_axis):
    """
    Simulate DMD diffraction using DFT. These produces peaks at a discrete set of frequencies which are
    (b-a)_x = wavelength / dx * ix / nx for ix = 0, ... nx - 1
    (b-a)_y = wavelength / dy * iy / ny for iy = 0, ... ny - 1
    these contain the full information of the output field. Intermediate values can be generated by (exact)
    interpolation using the DFT analog of the Shannon-Whittaker interpolation formula.

    :param pattern:
    :param efield_profile: illumination profile, which can include intensity and phase errors
    :param wavelength:
    :param gamma_on:
    :param gamma_off:
    :param dx:
    :param dy:
    :param wx:
    :param wy:
    :param uvec_in:
    :param order: (nx, ny)
    :param dn_orders: number of orders along nx and ny to compute around the central order of interest
    :param rot_axis_on:
    :param rot_axis_off:
    :return efields, sinc_efield_on, sinc_efield_off, b:
    """

    if isinstance(pattern, cp.ndarray):
        xp = cp
    else:
        xp = np

    efield_profile = xp.array(efield_profile)
    uvec_in = xp.array(uvec_in)

    ny, nx = pattern.shape

    # get allowed diffraction orders
    orders = np.stack(np.meshgrid(range(order[0] - dn_orders, order[0] + dn_orders + 1),
                                  range(order[1] - dn_orders, order[1] + dn_orders + 1)), axis=-1)

    order_xlims = [np.nanmin(orders[..., 0]), np.nanmax(orders[..., 0])]
    nx_orders = xp.arange(order_xlims[0], order_xlims[1] + 1)

    order_ylims = [np.nanmin(orders[..., 1]), np.nanmax(orders[..., 1])]
    ny_orders = xp.arange(order_ylims[0], order_ylims[1] + 1)

    # dft freqs
    fxs = xp.fft.fftshift(xp.fft.fftfreq(nx))
    fys = xp.fft.fftshift(xp.fft.fftfreq(ny))
    fxfx, fyfy = xp.meshgrid(fxs, fys)

    # to get effective frequencies, add diffraction orders
    # b_x = (b-a)_x + a_x
    uvecs_out_dft = xp.zeros((len(ny_orders) * ny, len(nx_orders) * nx, 3))
    uvecs_out_dft[..., 0] = (xp.tile(fxfx, [len(ny_orders), len(nx_orders)]) +
                             xp.kron(nx_orders, xp.ones((ny * len(nx_orders), nx)))) * wavelength / dx + \
                             uvec_in.squeeze()[0]
    # b_y = (b-a)_y + a_y
    uvecs_out_dft[..., 1] = (xp.tile(fyfy, [len(ny_orders), len(nx_orders)]) +
                             xp.kron(np.expand_dims(ny_orders, axis=1),
                                     xp.ones((ny, nx * len(ny_orders))))) * wavelength / dy + \
                             uvec_in.squeeze()[1]
    # b_z from normalization
    uvecs_out_dft[..., 2] = xp.sqrt(1 - uvecs_out_dft[..., 0] ** 2 - uvecs_out_dft[..., 1] ** 2)

    # get envelope functions for "on" and "off" states
    sinc_efield_on = wx * wy * blaze_envelope(wavelength, gamma_on, wx, wy,
                                              uvecs_out_dft - uvec_in,
                                              rot_axis_on)
    sinc_efield_off = wx * wy * blaze_envelope(wavelength, gamma_off, wx, wy,
                                               uvecs_out_dft - uvec_in,
                                               rot_axis_off)

    pattern_dft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(pattern * efield_profile)))
    pattern_complement_dft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift((1 - pattern) * efield_profile)))

    # efields = pattern_dft * sinc_efield_on + pattern_complement_dft * sinc_efield_off
    efields_on = xp.tile(pattern_dft, [len(nx_orders), len(ny_orders)]) * sinc_efield_on
    efields_off = xp.tile(pattern_complement_dft, [len(nx_orders), len(ny_orders)]) * sinc_efield_off
    efields = efields_on + efields_off

    return efields, pattern_dft, pattern_complement_dft, sinc_efield_on, sinc_efield_off, uvecs_out_dft


def interpolate_dmd_data(pattern: array,
                         efield_profile: array,
                         wavelength,
                         gamma_on: float,
                         gamma_off: float,
                         dx: float,
                         dy: float,
                         wx: float,
                         wy: float,
                         uvec_in,
                         order,
                         bvecs_interp,
                         rot_axis_on: Sequence[float],
                         rot_axis_off: Sequence[float]):
    """
    Exact interpolation of  dmd diffraction DFT data to other output angles using
    Shannon-Whittaker interpolation formula.

    todo: don't expect this to be any more efficient than simulate_dmd(), but should give the same result
    todo: possible way to speed up interpolation is with FT Fourier shift theorem. So approach would be to
    todo: choose certain shifts (e.g. make n-times denser and compute n^2 shift theorems)

    :param pattern:
    :param efield_profile:
    :param wavelength:
    :param gamma_on:
    :param gamma_off:
    :param dx:
    :param dy:
    :param wx:
    :param wy:
    :param uvec_in:
    :param order:
    :param bvecs_interp:
    :param rot_axis_on:
    :param rot_axis_off:
    :return efields:
    """

    if isinstance(pattern, cp.ndarray):
        xp = cp
    else:
        xp = np

    bvecs_interp = xp.atleast_2d(bvecs_interp)
    uvec_in = xp.atleast_2d(uvec_in)

    # get DFT results
    _, pattern_dft, pattern_dft_complement, _, _, bvec_dft = simulate_dmd_dft(pattern,
                                                                              efield_profile,
                                                                              wavelength,
                                                                              gamma_on,
                                                                              gamma_off,
                                                                              dx,
                                                                              dy,
                                                                              wx,
                                                                              wy,
                                                                              uvec_in,
                                                                              order,
                                                                              dn_orders=0,
                                                                              rot_axis_on=rot_axis_on,
                                                                              rot_axis_off=rot_axis_off)

    ny, nx = pattern.shape
    # dft freqs
    fxs = xp.fft.fftshift(xp.fft.fftfreq(nx))
    fys = xp.fft.fftshift(xp.fft.fftfreq(ny))

    bma = bvecs_interp - uvec_in
    sinc_efield_on = wx * wy * blaze_envelope(wavelength, gamma_on, wx, wy, bma, rot_axis_on)
    sinc_efield_off = wx * wy * blaze_envelope(wavelength, gamma_off, wx, wy, bma, rot_axis_off)

    def dft_interp_1d(d, v, n, frqs):
        arg = frqs - d / wavelength * v
        # val = 1 / n * np.sin(np.pi * arg * n) / np.sin(np.pi * arg) * np.exp(np.pi * 1j * arg * (n - 1))
        if xp.mod(n, 2) == 1:
            val = 1 / n * xp.sin(np.pi * arg * n) / xp.sin(np.pi * arg)
        else:
            val = 1 / n * xp.sin(np.pi * arg * n) / xp.sin(np.pi * arg) * xp.exp(-np.pi * 1j * arg)

        val[np.mod(np.round(arg, 14), 1) == 0] = 1
        return val

    nvecs = np.prod(bvecs_interp.shape[:-1])
    output_shape = bvecs_interp.shape[:-1]

    def calc(ii):
        ind = np.unravel_index(ii, output_shape)
        val = xp.sum((pattern_dft * sinc_efield_on[ind] + pattern_dft_complement * sinc_efield_off[ind]) *
                      xp.expand_dims(dft_interp_1d(dx, bma[ind][0], nx, fxs), axis=0) *
                      xp.expand_dims(dft_interp_1d(dy, bma[ind][1], ny, fys), axis=1))
        return val

    with ProgressBar():
        r = [dask.delayed(calc)(ii) for ii in range(nvecs)]
        results = dask.compute(*r)

    efields = xp.array(results).reshape(output_shape)

    return efields


def get_diffracted_power(pattern: array,
                         efield_profile: array,
                         wavelength: float,
                         gamma_on: float,
                         gamma_off: float,
                         dx: float,
                         dy: float,
                         wx: float,
                         wy: float,
                         uvec_in,
                         rot_axis_on: Sequence[float] = _dlp_1stgen_axis,
                         rot_axis_off: Sequence[float] = _dlp_1stgen_axis):
    """
    Compute input and output power.

    :param pattern:
    :param efield_profile:
    :param wavelength:
    :param gamma_on:
    :param gamma_off:
    :param dx:
    :param dy:
    :param wx:
    :param wy:
    :param uvec_in:
    :param rot_axis_on:
    :param rot_axis_off:
    :return power_in, power_out:
    """

    ny, nx = pattern.shape
    ax, ay, az = uvec_in.ravel()

    power_in = np.sum(np.abs(efield_profile)**2)

    _, pattern_dft, pattern_complement_dft, sinc_efield_on, sinc_efield_off, uvecs_out_dft = \
        simulate_dmd_dft(pattern,
                         efield_profile,
                         wavelength,
                         gamma_on,
                         gamma_off,
                         dx, dy, wx, wy,
                         uvec_in,
                         order=(0, 0),
                         dn_orders=0,
                         rot_axis_on=rot_axis_on,
                         rot_axis_off=rot_axis_off)

    # check that power is conserved here...
    assert np.abs(np.sum(np.abs(pattern_dft + pattern_complement_dft)**2) / (nx * ny) - power_in) < 1e-12

    # get FFT freqs
    fxs = (uvecs_out_dft[..., 0] - ax) * dx / wavelength
    fys = (uvecs_out_dft[..., 1] - ay) * dy / wavelength

    # get allowed diffraction orders
    ns, allowed_dc, allowed_any = get_physical_diff_orders(uvec_in, wavelength, dx, dy)

    def calc_power_order(order):
        ox, oy = order

        bxs = ax + wavelength / dx * (fxs + ox)
        bys = ay + wavelength / dy * (fys + oy)
        with np.errstate(invalid="ignore"):
            bzs = np.sqrt(1 - bxs ** 2 - bys ** 2)

        bvecs = np.stack((bxs, bys, bzs), axis=-1)
        bvecs[bxs ** 2 + bys ** 2 > 1] = np.nan

        envelope_on = blaze_envelope(wavelength, gamma_on, wx, wy, bvecs - uvec_in, rot_axis_on)
        envelope_off = blaze_envelope(wavelength, gamma_off, wx, wy, bvecs - uvec_in, rot_axis_off)

        on_sum = np.nansum(envelope_on ** 2)
        off_sum = np.nansum(envelope_off ** 2)

        pout = np.nansum(np.abs(envelope_on * pattern_dft + envelope_off * pattern_complement_dft) ** 2) / (nx * ny)

        return pout, on_sum, off_sum

    orders_x = ns[allowed_any, 0]
    orders_y = ns[allowed_any, 1]

    with ProgressBar():
        r = [dask.delayed(calc_power_order)((orders_x[ii], orders_y[ii])) for ii in range(len(orders_x))]
        results = dask.compute(*r)

    power_out_orders, on_sum_orders, off_sum_orders = zip(*results)
    power_out = np.sum(power_out_orders)
    # envelope_on_sum = np.sum(on_sum_orders)
    # envelope_off_sum = np.sum(off_sum_orders)

    return power_in, power_out


# ###########################################
# misc helper functions
# ###########################################
def sinc_fn(x: array) -> array:
    """
    Unnormalized sinc function, sinc(x) = sin(x) / x

    :param x:
    :return sinc(x):
    """
    if isinstance(x, cp.ndarray):
        xp = cp
    else:
        xp = np

    x = xp.atleast_1d(x)

    with np.errstate(divide='ignore', invalid='ignore'):
        y = xp.asarray(xp.sin(x) / x)
    y[x == 0] = 1

    return y


def get_rot_mat(rot_axis: Sequence[float],
                gamma: float) -> np.ndarray:
    """
    Get matrix which rotates points about the specified axis by the given angle. Think of this rotation matrix
    as acting on unit vectors, and hence its inverse R^{-1} transforms regular vectors. Therefore, we define
    this matrix such that it rotates unit vectors in a left-handed sense about the given axis for positive gamma.
    e.g. when rotating about the z-axis this becomes
    [[cos(gamma), -sin(gamma), 0],
     [sin(gamma), cos(gamma), 0],
     [0, 0, 1]]
    since vectors are acted on by the inverse matrix, they rotated in a right-handed sense about the given axis.

    :param rot_axis: unit vector specifying axis to rotate about, [nx, ny, nz]
    :param gamma: rotation angle in radians to transform point. A positive angle corresponds right-handed rotation
    about the given axis
    :return mat: 3x3 rotation matrix
    """
    if np.abs(np.linalg.norm(rot_axis) - 1) > 1e-12:
        raise ValueError("rot_axis must be a unit vector")

    nx, ny, nz = rot_axis
    mat = np.array([[nx**2 * (1 - np.cos(gamma)) + np.cos(gamma),
                     nx * ny * (1 - np.cos(gamma)) - nz * np.sin(gamma),
                     nx * nz * (1 - np.cos(gamma)) + ny * np.sin(gamma)],
                    [nx * ny * (1 - np.cos(gamma)) + nz * np.sin(gamma),
                     ny**2 * (1 - np.cos(gamma)) + np.cos(gamma),
                     ny * nz * (1 - np.cos(gamma)) - nx * np.sin(gamma)],
                    [nx * nz * (1 - np.cos(gamma)) - ny * np.sin(gamma),
                     ny * nz * (1 - np.cos(gamma)) + nx * np.sin(gamma),
                     nz**2 * (1 - np.cos(gamma)) + np.cos(gamma)]])

    return mat


def get_rot_mat_angle_axis(rot_mat: np.ndarray) -> (np.ndarray, float):
    """
    Given a rotation matrix, determine the axis it rotates about and the angle it rotates through. This is
    the inverse function for get_rot_mat()

    Note that get_rot_mat_angle_axis(get_rot_mat(axis, angle)) can return either axis, angle or -axis, -angle
    as these two rotation matrices are equivalent

    :param rot_mat:
    :return rot_axis, angle:
    """
    if np.linalg.norm(rot_mat.dot(rot_mat.transpose()) - np.identity(rot_mat.shape[0])) > 1e-12:
        raise ValueError("rot_mat was not a valid rotation matrix")

    eig_vals, eig_vects = np.linalg.eig(rot_mat)

    # rotation matrix must have one eigenvalue that is 1 to numerical precision
    ind = np.argmin(np.abs(eig_vals - 1))

    # construct basis with e3 = rotation axis
    e3 = eig_vects[:, ind].real

    if np.linalg.norm(np.cross(np.array([0, 1, 0]), e3)) != 0:
        e1 = np.cross(np.array([0, 1, 0]), e3)
    else:
        e1 = np.cross(np.array([1, 0, 0]), e3)
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(e3, e1)

    # basis change matrix to look like rotation about z-axis
    mat_basis_change = np.vstack((e1, e2, e3)).transpose()

    # transformed rotation matrix
    r_bc = np.linalg.inv(mat_basis_change).dot(rot_mat.dot(mat_basis_change))
    angle = np.arcsin(r_bc[1, 0]).real
    
    # the pairs (e3, angle) and (-e3, -angle) represent the same matrix
    # choose the pair so that the largest component of e3 is positive
    ind_max = np.argmax(np.abs(e3))
    if e3[ind_max] < 0:
        e3 = -e3
        angle = -angle

    return e3, angle


# ###########################################
# convert between coordinate systems
# ###########################################
def xyz2mirror(vx: array,
               vy: array,
               vz: array,
               gamma: float,
               rot_axis: Sequence[float] = _dlp_1stgen_axis) -> (array, array, array):
    """
    Convert vector with components vx, vy, vz to v1, v2, v3.

    The unit vectors ex, ey, ez are defined along the axes of the DMD body,
    whereas the unit vectors e1, e2, e3 are given by
    e1 = (ex - ey) / sqrt(2) * cos(gamma) - ez * sin(gamma)
    e2 = (ex + ey) / sqrt(2)
    e3 = (ex - ey) / sqrt(2) sin(gamma) + ez * cos(gamma)
    which are convenient because e1 points along the direction the micromirrors swivel and
    e3 is normal to the DMD micrmirrors

    :param vx:
    :param vy:
    :param vz:
    :param gamma:
    :param rot_axis:
    :return: v1, v2, v3
    """
    rot_mat = get_rot_mat(rot_axis, gamma)

    # v_{123} = R^{-1} * v_{xyz}
    # v1 = e1 \cdot v = vx * e1 \cdot ex + vy * e1 \cdot ey + vz * e1 \cdot ez
    v1 = vx * rot_mat[0, 0] + vy * rot_mat[1, 0] + vz * rot_mat[2, 0]
    v2 = vx * rot_mat[0, 1] + vy * rot_mat[1, 1] + vz * rot_mat[2, 1]
    v3 = vx * rot_mat[0, 2] + vy * rot_mat[1, 2] + vz * rot_mat[2, 2]

    return v1, v2, v3


def mirror2xyz(v1: array,
               v2: array,
               v3: array,
               gamma: float,
               rot_axis: Sequence[float] = _dlp_1stgen_axis) -> (array, array, array):
    """
    Inverse function for xyz2mirror()

    :param v1:
    :param v2:
    :param v3:
    :param gamma:
    :param rot_axis:
    :return:
    """
    rot_mat = get_rot_mat(rot_axis, gamma)

    # v_{xyz} = R * v_{123}
    # vx = ex \cdot v = v1 * ex \cdot e1 + v2 * ex \cdot e2 + v3 * ex \cdot e3
    vx = v1 * rot_mat[0, 0] + v2 * rot_mat[0, 1] + v3 * rot_mat[0, 2]
    vy = v1 * rot_mat[1, 0] + v2 * rot_mat[1, 1] + v3 * rot_mat[1, 2]
    vz = v1 * rot_mat[2, 0] + v2 * rot_mat[2, 1] + v3 * rot_mat[2, 2]

    return vx, vy, vz


def xyz2mpz(vx: array,
            vy: array,
            vz: array) -> (array, array, array):
    """
    Convert from x, y, z coordinate system to m = (x-y)/sqrt(2), p = (x+y)/sqrt(2), z

    :param vx:
    :param vy:
    :param vz:
    :return vm, vp, vz:
    """
    vp = np.array(vx + vy) / np.sqrt(2)
    vm = np.array(vx - vy) / np.sqrt(2)
    vz = np.array(vz, copy=True)

    return vm, vp, vz


def mpz2xyz(vm: array,
            vp: array,
            vz: array) -> (array, array, array):
    """
    Convert from m = (x-y)/sqrt(2), p = (x+y)/sqrt(2), z coordinate system to x, y, z
    :param vm:
    :param vp:
    :param vz:
    :return, vx, vy, vz:
    """
    vx = np.array(vm + vp) / np.sqrt(2)
    vy = np.array(vp - vm) / np.sqrt(2)
    vz = np.array(vz, copy=True)

    return vx, vy, vz


# ###########################################
# convert between different angular or unit vector representations of input and output directions
# ###########################################
def angle2xy(tp: array,
             tm: array) -> (array, array):
    """
    Convert angle projections along the x- and y-axis to angle projections along the p=(x+y)/sqrt(2)
    and m=(x-y)/sqrt(2) axis.

    :param tp:
    :param tm:
    :return tx, ty:
    """

    tx = np.arctan((np.tan(tp) + np.tan(tm)) / np.sqrt(2))
    ty = np.arctan((np.tan(tp) - np.tan(tm)) / np.sqrt(2))

    return tx, ty


def angle2pm(tx: array,
             ty: array) -> (array, array):
    """
    Convert angle projections along the p=(x+y)/sqrt(2) and m=(x-y)/sqrt(2) to x and y axes.

    :param tx:
    :param ty:
    :return tp, tm:
    """

    tm = np.arctan((np.tan(tx) - np.tan(ty)) / np.sqrt(2))
    tp = np.arctan((np.tan(tx) + np.tan(ty)) / np.sqrt(2))

    return tp, tm


def uvector2txty(vx: array,
                 vy: array,
                 vz: array) -> (array, array):
    """
    Convert unit vector from components to theta_x, theta_y representation. Inverse function for get_unit_vector()

    NOTE: tx and ty are defined differently depending on the sign of the z-component of the unit vector
    :param vx:
    :param vy:
    :param vz:
    :return:
    """
    norm_factor = np.abs(1 / vz)
    tx = np.arctan(vx * norm_factor)
    ty = np.arctan(vy * norm_factor)

    return tx, ty


def uvector2tmtp(vx: array,
                 vy: array,
                 vz: array) -> (array, array):
    """
    Convert unit vector to angle projections along ep and em
    :param vx:
    :param vy:
    :param vz:
    :return tp, tm:
    """
    tx, ty = uvector2txty(vx, vy, vz)
    tp, tm = angle2pm(tx, ty)
    return tp, tm


def pm2uvector(tm: array,
               tp: array,
               incoming: bool) -> (array, array, array):
    """

    :param tm:
    :param tp:
    :param incoming:
    :return:
    """
    tx, ty = angle2xy(tp, tm)
    return xy2uvector(tx, ty, incoming)


def xy2uvector(tx: array,
               ty: array,
               incoming: bool) -> (array, array, array):
    """
    Get incoming or outgoing unit vector of light propagation parameterized by angles tx and ty

    Let a represent an incoming vector, and b and outgoing one. We parameterize these by
    a = az * [tan(tx_a), tan(ty_a), -1]
    b = |bz| * [tan(tb_x), tan(tb_y), 1]
    choosing negative z component for outgoing vectors is effectively taking a different
    conventions for the angle between b and the z axis (compared with a and
    the z-axis). We do this so that e.g. the law of reflection would give
    theta_a = theta_b, instead of theta_a = -theta_b, which would hold if we
    defined everything symmetrically.

    :param tx: arbitrary size
    :param ty: same size as tx
    :param incoming: true if representing a vector pointing in the negative z-direction, i.e.
      incident on the DMD
    :return uvec: unit vectors, array of size tx.size x 3
    """
    tx = np.atleast_1d(tx)
    ty = np.atleast_1d(ty)
    norm = np.sqrt(np.tan(tx)**2 + np.tan(ty)**2 + 1)
    if incoming:
        ux = np.tan(tx)
        uy = np.tan(ty)
        uz = -np.ones(tx.shape)
    else:
        ux = np.tan(tx)
        uy = np.tan(ty)
        uz = np.ones(tx.shape)

    uvec = np.stack((ux, uy, uz), axis=-1) / np.expand_dims(norm, axis=-1)

    return uvec


# ###########################################
# diffraction directions for different pattern frequencies
# ###########################################
def dmd_frq2uvec(uvec_out_dc: array,
                 fx: float,
                 fy: float,
                 wavelength: float,
                 dx: float,
                 dy: float) -> (array, array, array):
    """
    Determine the output diffraction vector b(f) given the output vector b(0) and the
    spatial frequency f = [fx, fy] in 1/mirrors.

    :param uvec_out_dc: main diffraction output unit vector, i.e. DC diffraction component output direction
    :param fx: 1/mirror
    :param fy: 1/mirror
    :param wavelength: distance units
    :param dx: same units as wavelength
    :param dy: same units as wavelength
    :return bfx, bfy, bfz:
    """
    uvec_out_dc = np.squeeze(uvec_out_dc)

    bfx = uvec_out_dc[0] + wavelength / dx * fx
    bfy = uvec_out_dc[1] + wavelength / dy * fy
    bfz = np.sqrt(1 - bfx**2 - bfy**2)

    return bfx, bfy, bfz


def uvec2dmd_frq(uvec_out_dc: Sequence[float],
                 uvec_f: array,
                 wavelength: float,
                 dx: float,
                 dy: float) -> (array, array):
    """
    Inverse function of freq2uvec

    :param uvec_out_dc:
    :param uvec_f:
    :param wavelength:
    :param dx:
    :param dy:
    :return fx, fy:
    """
    fx = (uvec_f[..., 0] - uvec_out_dc[0]) * dx / wavelength
    fy = (uvec_f[..., 1] - uvec_out_dc[1]) * dy / wavelength
    return fx, fy


# ###########################################
# mapping from DMD coordinates to optical axis coordinates
# ###########################################
def get_fourier_plane_basis(optical_axis_uvec: Sequence[float]):
    """
    Get basis vectors which are orthogonal to a given optical axis. This is useful when
    we suppose that a lens has been placed one focal length after the DMD and we are interested
    in computing the optical field in the back focal plane of the lens (i.e. the Fourier plane) or
    determining the relative angles between diffraction directions and the optical axis.

    This basis is chosen such that xb would point along the x-axis and yb would point
    along the y-axis if optical_axis_uvec = (0, 0, 1).

    :param optical_axis_uvec: unit vector defining the optical axis
    :return xb, yb: (xb, yb, optical_axis_uvec) form a right-handed coordinate system
    """
    xb = np.array([optical_axis_uvec[2], 0, -optical_axis_uvec[0]]) / np.sqrt(optical_axis_uvec[0] ** 2 + optical_axis_uvec[2] ** 2)
    yb = np.cross(optical_axis_uvec, xb)

    return xb, yb


def dmd_frq2opt_axis_uvec(fx: array,
                          fy: array,
                          bvec,
                          opt_axis_vec,
                          dx: float,
                          dy: float,
                          wavelength: float) -> (array, array, array):
    """
    Convert from DMD pattern frequencies to unit vectors about the optical axis. This can be easily converted to
    either spatial frequencies or positions in the Fourier plane.
    fx = b_xp / wavelength
    x_fourier_plane = b_xp * focal_len

    :param fx: 1/mirror
    :param fy: 1/mirror
    :param bvec: main diffraction order output angle, which is the angle a flat pattern (i.e. a pattern of
    frequency fx=0, fy=0) is diffracted into.
    :param opt_axis_vec: unit vector pointing along the optical axis of the Fourier plane
    :param dx: DMD pitch
    :param dy: DMD pitch
    :param wavelength: same units as DMD pitch

    :return bf_xp, bf_yp, bf_zp: vector components in the pupil plane and along the optical axis. In most cases
    bf_zp is not useful. But bf_xp and bf_yp may be converted to pupil spatial coordinates by multiplying them
    with the lens focal length.
    """
    if np.abs(np.linalg.norm(bvec) - 1) > 1e-12:
        raise ValueError("bvec was not a unit vector")

    if np.abs(np.linalg.norm(opt_axis_vec) - 1) > 1e-12:
        raise ValueError("pvec was not a unit vector")

    fx = np.atleast_1d(fx)
    fy = np.atleast_1d(fy)

    bf_xs, bf_ys, bf_zs = dmd_frq2uvec(bvec, fx, fy, wavelength, dx, dy)

    # optical axis basis
    xp, yp = get_fourier_plane_basis(opt_axis_vec)

    # convert bfs to pupil coordinates
    # bf_xp = b(f) \dot x_p = bx * x \dot x_p + by * y \dot y_p + bz * z \dot z_p
    bf_xp = bf_xs * xp[0] + bf_ys * xp[1] + bf_zs * xp[2]
    bf_yp = bf_xs * yp[0] + bf_ys * yp[1] + bf_zs * yp[2]
    bf_zp = bf_xs * opt_axis_vec[0] + bf_ys * opt_axis_vec[1] + bf_zs * opt_axis_vec[2]

    # note that there many other ways of thinking about this problem.
    # another natural way is to being with b(f) and zp. Construct an orthogonal coordinate system with
    # v2 = zp \cross b(f) / norm = (b_xp * yp - b_yp * xp) / sqrt(b_xp**2 + b_yp**2)
    # v1 = v2 \cross zp / norm = (b_xp * xp + b_yp * yp) / sqrt(b_xp**2 + b_yp**2)
    # then the position in the pupil plane is
    # r = v1 * fl * sin(theta) = (b_xp * xp + b_yp * yp) * fl
    # which is just what we get from the above...

    return bf_xp, bf_yp, bf_zp


def dmd_uvec2opt_axis_uvec(dmd_uvecs: array,
                           opt_axis_vec: Sequence[float]) -> (array, array, array):
    """
    Convert unit vectors expressed relative to the dmd coordinate system
    dmd_uvecs = bx * ex + by * ey + bz * ez
    to expression relative to the optical axis coordinate system
    opt_axis_uvecs = bxp * exp + byp * eyp + bzp * ezp
    For refractive index n = 1, can also be thought of as freq * wavelength

    :param dmd_uvecs:
    :param opt_axis_vec:
    :return bf_xp, bf_yp, bf_zp:
    """
    dmd_uvecs = np.atleast_2d(dmd_uvecs)

    # optical axis basis
    xp, yp = get_fourier_plane_basis(opt_axis_vec)

    # convert bfs to pupil coordinates
    bf_xs = dmd_uvecs[..., 0]
    bf_ys = dmd_uvecs[..., 1]
    bf_zs = dmd_uvecs[..., 2]
    # bf_xp = b(f) \dot x_p = bx * x \dot x_p + by * y \dot y_p + bz * z \dot z_p
    bf_xp = bf_xs * xp[0] + bf_ys * xp[1] + bf_zs * xp[2]
    bf_yp = bf_xs * yp[0] + bf_ys * yp[1] + bf_zs * yp[2]
    bf_zp = bf_xs * opt_axis_vec[0] + bf_ys * opt_axis_vec[1] + bf_zs * opt_axis_vec[2]

    return bf_xp, bf_yp, bf_zp


def opt_axis_uvec2dmd_uvec(opt_axis_uvecs: np.ndarray,
                           opt_axis_vec) -> (array, array, array):
    """
    Convert unit vectors expressed relative to the optical axis coordinate system,
    opt_axis_uvecs = bxp * exp + byp * eyp + bzp * ezp
    to expression relative to the DMD coordinate system
    dmd_uvecs = bx * ex + by * ey + bz * ez

    :param opt_axis_uvecs: (bxp, byp, bzp)
    :param opt_axis_vec: (ox, oy, oz)
    :return bx, by, bz:
    """
    # optical axis basis
    xp, yp = get_fourier_plane_basis(opt_axis_vec)

    # opt_axis_uvecs = (x_oa, y_oa, z_oa)
    bx = opt_axis_uvecs[..., 0] * xp[0] + opt_axis_uvecs[..., 1] * yp[0] + opt_axis_uvecs[..., 2] * opt_axis_vec[0]
    by = opt_axis_uvecs[..., 0] * xp[1] + opt_axis_uvecs[..., 1] * yp[1] + opt_axis_uvecs[..., 2] * opt_axis_vec[1]
    bz = opt_axis_uvecs[..., 0] * xp[2] + opt_axis_uvecs[..., 1] * yp[2] + opt_axis_uvecs[..., 2] * opt_axis_vec[2]

    return bx, by, bz


# ###########################################
# functions for blaze condition only
# ###########################################
def blaze_envelope(wavelength: float,
                   gamma: float,
                   wx: float,
                   wy: float,
                   b_minus_a: array,
                   rot_axis: Sequence[float] = _dlp_1stgen_axis) -> array:
    """
    Compute normalized blaze envelope function. Envelope function has value 1 where the blaze condition is satisfied.
    This is the result of doing the integral
    envelope(b-a) = \int ds dt exp[ ik Rn*(s,t,0) \cdot (a-b)] / w**2
    = \int ds dt exp[ ik * (A_+*s + A_-*t)] / w**2
    = sinc(0.5 * k * w * A_+) * sinc(0.5 * k * w * A_-)

    The overall electric field is given by
    E(b-a) = (diffraction from mirror pattern) x envelope(b-a)

    :param wavelength: wavelength of light. Units are arbitrary, but must be the same for wavelength, wx, and wy
    :param gamma: mirror swivel angle, in radians
    :param wx: mirror width in x-direction. Same units as wavelength.
    :param wy: mirror width in y-direction. Same units as wavelength.
    :param b_minus_a: array of size no x ... x nk x 3 difference between output (b) and input (a) unit vectors.
    :param rot_axis: unit vector about which the mirror swivels. Typically (1, 1, 0) / np.sqrt(2)
    :return envelope: same length as b_minus_a
    """

    k = 2*np.pi / wavelength
    val_plus, val_minus = blaze_condition_fn(gamma, b_minus_a, rot_axis=rot_axis)
    envelope = sinc_fn(0.5 * k * wx * val_plus) * sinc_fn(0.5 * k * wy * val_minus)
    return envelope


def blaze_condition_fn(gamma: float,
                       b_minus_a: array,
                       rot_axis: Sequence[float] = _dlp_1stgen_axis) -> (array, array):
    """
    Return the dimensionsless part of the sinc function argument which determines the blaze condition.
    We refer to these functions as A_+(b-a, gamma) and A_-(b-a, gamma).

    These are related to the overall electric field by
    E(b-a) = (diffraction from mirror pattern) x w**2 * sinc(0.5 * k * w * A_+) * sinc(0.5 * k * w * A_-)

    :param gamma: angle micro-mirror normal makes with device normal
    :param b_minus_a: outgoing unit vector - incoming unit vector. array of shape n0 x n1 x ... x 3
    :param rot_axis: unit vector about which the mirror swivels. Typically use (1, 1, 0) / np.sqrt(2)
    :return val: A_+ or A_-, depending on the mode
    """

    rot_mat = get_rot_mat(rot_axis, gamma)
    # phase(s, t) = R.dot([s, t, 0]) \cdot (b-a) =
    # (R[0, 0] * vx + R[1, 0] * vy + R[2, 0] * vz) * s +
    # (R[0, 1] * vx + R[1, 1] * vy + R[2, 1] * vz) * t

    val_plus = -rot_mat[0, 0] * b_minus_a[..., 0] + \
               -rot_mat[1, 0] * b_minus_a[..., 1] + \
               -rot_mat[2, 0] * b_minus_a[..., 2]
    val_minus = -rot_mat[0, 1] * b_minus_a[..., 0] + \
                -rot_mat[1, 1] * b_minus_a[..., 1] + \
                -rot_mat[2, 1] * b_minus_a[..., 2]

    return val_plus, val_minus


def solve_blaze_output(uvecs_in: array,
                       gamma: float,
                       rot_axis: Sequence[float] = _dlp_1stgen_axis) -> array:
    """
    Find the output angle which satisfies the blaze condition for arbitrary input angle

    :param uvecs_in: N x 3 array of unit vectors (ax, ay, az)
    :param gamma: DMD mirror angle in radians
    :param rot_axis:
    :return uvecs_out: unit vectors giving output directions
    """

    uvecs_in = np.atleast_2d(uvecs_in)
    # convert to convenient coordinates and apply blaze
    a1, a2, a3 = xyz2mirror(uvecs_in[..., 0],
                            uvecs_in[..., 1],
                            uvecs_in[..., 2],
                            gamma,
                            rot_axis)
    bx, by, bz = mirror2xyz(a1, a2, -a3, gamma, rot_axis)
    uvecs_out = np.stack((bx, by, bz), axis=-1)

    return uvecs_out


def solve_blaze_input(uvecs_out: array,
                      gamma: float,
                      rot_axis: Sequence[float] = _dlp_1stgen_axis) -> array:
    """
    Find the input angle which satisfies the blaze condition for arbitrary output angle.

    :param uvecs_out:
    :param gamma:
    :param rot_axis:
    :return uvecs_in:
    """
    uvecs_in = solve_blaze_output(uvecs_out, gamma, rot_axis)
    return uvecs_in


# ###########################################
# functions for diffraction conditions only
# ###########################################
def get_physical_diff_orders(uvec_in,
                             wavelength: float,
                             dx: float,
                             dy: float):
    """
    Determine which diffraction orders are physically supported by the grating given a certain input direction

    :param uvec_in:
    :param wavelength:
    :param dx:
    :param dy:
    :return ns, allowed_dc, allowed_any: n x n x 2 array, where ns[ii, jj] = np.array([nx[ii, jj], ny[ii, jj]).
      allowed_dc and allowed_any are boolean arrays which indicate which ns have forbidden DC values and which ns
      have all forbidden diffraction orders
    """
    ax, ay, az = uvec_in.ravel()

    nx_max = int(np.floor(dx / wavelength * (1 - ax))) + 1
    nx_min = int(np.ceil(dx / wavelength * (-1 - ax))) - 1
    ny_max = int(np.floor(dy / wavelength * (1 - ay))) + 1
    ny_min = int(np.ceil(dy / wavelength * (-1 - ay))) - 1

    nxnx, nyny = np.meshgrid(range(nx_min, nx_max + 1), range(ny_min, ny_max + 1))
    nxnx = nxnx.astype(float)
    nyny = nyny.astype(float)

    # check which DC orders are allowed
    bx = ax + wavelength/dx * nxnx
    by = ay + wavelength/dy * nyny
    allowed_dc = bx**2 + by**2 <= 1

    # check corner diffraction orders
    bx_c1 = ax + wavelength / dx * (nxnx + 0.5)
    by_c1 = ay + wavelength / dy * (nyny + 0.5)
    bx_c2 = ax + wavelength / dx * (nxnx + 0.5)
    by_c2 = ay + wavelength / dy * (nyny - 0.5)
    bx_c3 = ax + wavelength / dx * (nxnx - 0.5)
    by_c3 = ay + wavelength / dy * (nyny + 0.5)
    bx_c4 = ax + wavelength / dx * (nxnx - 0.5)
    by_c4 = ay + wavelength / dy * (nyny - 0.5)
    allowed_any = np.logical_or.reduce((bx_c1**2 + by_c1**2 <= 1,
                                        bx_c2**2 + by_c2**2 <= 1,
                                        bx_c3**2 + by_c3**2 <= 1,
                                        bx_c4**2 + by_c4**2 <= 1))

    ns = np.stack((nxnx, nyny), axis=-1)

    return ns, allowed_dc, allowed_any


def find_nearst_diff_order(uvec_in,
                           uvec_out,
                           wavelength: float,
                           dx: float,
                           dy: float):
    """
    Given an input and output direction, find the nearest diffraction order

    :param uvec_in:
    :param uvec_out:
    :param wavelength:
    :param dx:
    :param dy:
    :return:
    """
    ns, allowed_dc, _ = get_physical_diff_orders(uvec_in, wavelength, dx, dy)
    ns[np.logical_not(allowed_dc)] = np.nan

    ux, uy, uz = uvec_out.ravel()
    ax, ay, az = uvec_in.ravel()

    bxs = ax + ns[..., 0] * wavelength / dx
    bys = ay + ns[..., 1] * wavelength / dy
    bzs = np.sqrt(1 - bxs**2 - bys**2)

    dists = np.sqrt((bxs - ux)**2 + (bys - uy)**2 + (bzs - uz)**2)
    ind_min = np.unravel_index(np.nanargmin(dists), ns[..., 0].shape)

    order = ns[ind_min].astype(int)

    return order


def solve_diffraction_input(uvecs_out: array,
                            dx: float,
                            dy: float,
                            wavelength: float,
                            order: Sequence[int]) -> array:
    """
    Solve for the input direction which will be diffracted into the given output direction by
    the given diffraction order of the DMD

    :param uvecs_out:
    :param dx:
    :param dy:
    :param wavelength:
    :param order: (order_x, order_y)
    :return avecs:
    """
    uvecs_out = np.atleast_2d(uvecs_out)

    ax = uvecs_out[..., 0] - wavelength / dx * order[0]
    ay = uvecs_out[..., 1] - wavelength / dy * order[1]
    az = -np.sqrt(1 - ax**2 - ay**2)
    uvecs_in = np.stack((ax, ay, az), axis=-1)

    return uvecs_in


def solve_diffraction_output(uvecs_in: array,
                             dx: float,
                             dy: float,
                             wavelength: float,
                             nx: Union[array, int],
                             ny: Union[array, int]) -> array:
    """
    Solve for the output direction into which the given input direction will be
    diffracted by the given order of the DMD

    The diffraction condition is:
    bx - ax = wavelength / d * nx
    by - ay = wavelength / d * ny

    :param uvecs_in: n0 x n1 x ... x nk x 3
    :param dx: x-dmd pitch
    :param dy: y-dmd pitch
    :param wavelength:
    :param nx: x-diffraction orders. Either integer, or shape should be compatible
      to broadcast with array of size no x n1 x ... x nk
    :param ny: y-diffraction orders
    :return uvecs_out:
    """
    uvecs_in = np.atleast_2d(uvecs_in)

    bx = uvecs_in[..., 0] + wavelength / dx * nx
    by = uvecs_in[..., 1] + wavelength / dy * ny
    with np.errstate(invalid="ignore"):
        bz = np.sqrt(1 - bx**2 - by**2)

    # these points have no solution
    is_nan = np.isnan(bz)
    bx[is_nan] = np.nan
    by[is_nan] = np.nan

    uvecs_out = np.stack((bx, by, bz), axis=-1)

    return uvecs_out


# ###########################################
# functions for solving blaze + diffraction conditions
# ###########################################
def get_diffraction_order_limits(wavelength: float,
                                 d: float,
                                 gamma: float,
                                 rot_axis: Sequence[float] = _dlp_1stgen_axis):
    """
    Find the maximum and minimum diffraction orders consistent with given parameters and the blaze condition.
    Note that only diffraction orders of the form (n, -n) can satisfy the Blaze condition, hence only the value
    n is returned and not a 2D diffraction order tuple.

    # todo: only gives results if mirror swivels along (x+y) axis

    :param wavelength: wavelength of light
    :param d: mirror pitch (in same units as wavelength)
    :param gamma: mirror angle
    :param rot_axis:
    :return nmax, nmin: maximum and minimum indices of diffraction order
    """

    if np.linalg.norm(np.array(rot_axis) - np.array(_dlp_1stgen_axis)) > 1e-12:
        raise NotImplementedError("get_diffraction_order_limits() not get implemented for arbitrary rotation axis")
    rot_mat = get_rot_mat(rot_axis, gamma)

    # # solution for maximum order
    if rot_mat[0, 2] <= 0:
        nmax = 0
        nmin = int(np.ceil(-d / wavelength * 2 * rot_mat[0, 2]))
    else:
        nmax = int(np.floor(d / wavelength * 2 * rot_mat[0, 2]))
        nmin = 0

    return np.array([nmin, nmax], dtype=int)


def solve_1color_1d(wavelength: float,
                    d: float,
                    gamma: float,
                    order: int) -> (array, array):
    """
    Solve for the input and output angles satisfying both the diffraction condition and blaze angle for a given
    diffraction order (if possible). These function assumes that (1) the mirror rotation axis is the (x+y) axis and
    (2) the input and output beams are in the x-y plane.

    The two conditions to be solved are
    (1) theta_in - theta_out = 2*gamma
    (2) sin(theta_in) - sin(theta_out) = sqrt(2) * wavelength / d * n

    This function is a wrapper for solve_combined_condition() simplified for the 1D geometry.

    :param wavelength: wavelength of light
    :param d: mirror pitch (in same units as wavelength)
    :param gamma: angle mirror normal makes with DMD body normal
    :param order: diffraction order index. Full order index is (nx, ny) = (order, -order)

    :return uvecs_in: list of input angle solutions as unit vectors
    :return uvecs_out: list of output angle solutions as unit vectors
    """
    # uvec_fn, _ = solve_combined_condition(d, gamma, wavelength, order, rot_axis=(1/np.sqrt(2), 1/np.sqrt(2), 0))
    # 1D solutions are the solutions where a_{x+y} = 0
    # this implies a1 = -a2

    a3 = -1 / np.sqrt(2) / np.sin(gamma) * wavelength / d * order
    a1_p = np.sqrt(1 - a3**2) / np.sqrt(2)
    a2_p = - a1_p

    a1_m = -np.sqrt(1 - a3**2) / np.sqrt(2)
    a2_m = -a1_m

    a_p = mirror2xyz(a1_p, a2_p, a3, gamma, rot_axis=_dlp_1stgen_axis)
    a_m = mirror2xyz(a1_m, a2_m, a3, gamma, rot_axis=_dlp_1stgen_axis)
    b_p = mirror2xyz(a1_p, a2_p, -a3, gamma, rot_axis=_dlp_1stgen_axis)
    b_m = mirror2xyz(a1_m, a2_m, -a3, gamma, rot_axis=_dlp_1stgen_axis)

    uvecs_in = np.vstack((a_p, a_m))
    uvecs_out = np.vstack((b_p, b_m))
    return uvecs_in, uvecs_out


def solve_2color_on_off(d: float,
                        gamma_on: float,
                        wavelength_on: float,
                        n_on: int,
                        wavelength_off: float,
                        n_off: int):
    """
    Solve the combined blaze and diffraction conditions jointly for two wavelengths, assuming the first wavelength
    couples from the DMD "on" mirrors and the second from the "off" mirrors.

    :param d: mirror pitch
    :param gamma_on: mirror angle in ON state in radians. Assume that gamma_off = -gamma_on
    :param wavelength_on: wavelength of light incident on ON mirrors. Must be in same units as d
    :param n_on: diffraction order for ON mirrors
    :param wavelength_off: wavelength of light incident on OFF mirrors. Must be in same units as d
    :param n_off: diffraction order for OFF mirrors

    :return b_vecs: output unit vectors. Two solution vectors, size 2 x 3
    :return a_vecs_on: input unit vectors for ON mirrors
    :return b_vecs_on: input unit vectors for OFF mirrors
    """

    b3_on = -1 / np.sqrt(2) / np.sin(gamma_on) * wavelength_on / d * n_on
    b3_off = 1 / np.sqrt(2) / np.sin(gamma_on) * wavelength_off / d * n_off

    # equate b_on and b_off, and solve for bz, bx, by
    # (1) b3_on + b3_off = 2 * cos(gamma) * bz
    # (2) b3_on - b3_off = np.sqrt(2) * np.sin(gamma) * (bx - by)
    bz = 0.5 / np.cos(gamma_on) * (b3_on + b3_off)

    # quadratic equation for bx from (2)
    c1 = 1
    c2 = -(b3_on - b3_off) / np.sqrt(2) / np.sin(gamma_on)
    c3 = 0.5 * (bz**2 + (b3_on - b3_off)**2 / 2 / np.sin(gamma_on)**2 - 1)

    bxs = np.array([0.5 * (-c2 + np.sqrt(c2**2 - 4 * c3)) / c1,
                    0.5 * (-c2 - np.sqrt(c2**2 - 4 * c3)) / c1])

    # apply eq. (2) again to get by (since lost information when we squared it to get quadratic eqn)
    bys = bxs - (b3_on - b3_off) / np.sqrt(2) / np.sin(gamma_on)

    # assemble b-vector
    b_vecs = np.array([[bxs[0], bys[0], bz], [bxs[1], bys[1], bz]])

    # todo: get rid of loop
    for ii in range(b_vecs.shape[0]):
        if np.any(np.isnan(b_vecs[ii])):
            b_vecs[ii, :] = np.nan

    # get input unit vectors
    b1_on, b2_on, b3_on = xyz2mirror(b_vecs[..., 0],
                                     b_vecs[..., 1],
                                     b_vecs[..., 2],
                                     gamma_on)

    a_vecs_on = np.stack(mirror2xyz(b1_on,
                                    b2_on,
                                    -b3_on,
                                    gamma_on),
                         axis=-1)

    b1_off, b2_off, b3_off = xyz2mirror(b_vecs[..., 0],
                                        b_vecs[..., 1],
                                        b_vecs[..., 2],
                                        -gamma_on)

    a_vecs_off = np.stack(mirror2xyz(b1_off,
                                     b2_off,
                                     -b3_off,
                                     -gamma_on),
                          axis=-1)

    return b_vecs, a_vecs_on, a_vecs_off


def solve_combined_condition(d: float,
                             gamma: float,
                             rot_axis: Sequence[float],
                             wavelength: float,
                             order: Sequence[int]) -> (Callable, Callable):
    """
    Solve the combined diffraction/blaze condition. Since in general the system of equation is overdetermined,
    return the vectors which solve the diffraction condition and minimize the blaze condition cost, defined by
    C(a, b) = (b1-a1)^2 + (b2-a2)^2
    Since this optimization problem is underdetermined, as it only depends on b-a, return callables which solve it
    for specific values of ax or ay

    :param d: DMD mirror pitch
    :param gamma: DMD mirror angle along the x-y direction in radians
    :param rot_axis:
    :param wavelength: wavelength in same units as DMD mirror pitch
    :param order: (nx, ny) = (order, -order)
    :return ab_from_ax: function which returns the full vectors a(ax), b(ax). If ax is a vector of length N,
      then this will return two vectors of size 2 x N x 3, since there are two solutions for eaach ax
    :return ab_from_ay: function which return the full vectors a(ay), b(ay)
    """

    nx, ny = order
    rot_mat = get_rot_mat(rot_axis, gamma)

    # differences between vectors b and a
    # this condition derived from Lagrange multiplier method
    bma_z = -wavelength / d * (nx * (rot_mat[2, 0] * rot_mat[0, 0] + rot_mat[2, 1] * rot_mat[0, 1]) +
                               ny * (rot_mat[2, 0] * rot_mat[1, 0] + rot_mat[2, 1] * rot_mat[1, 1])) / \
            (rot_mat[2, 0]**2 + rot_mat[2, 1]**2)

    bma_norm = np.sqrt((nx * wavelength / d)**2 + (ny * wavelength / d)**2 + bma_z**2)
    b_dot_a = 0.5 * (2 - bma_norm**2)

    # choose value for ax, then use ax*bx + ay*by - sqrt(1 - ax^2 - ay^2) * sqrt(1 - bx^2 - by^2) = b_dot_a
    # together with diffraction condition to obtain quadratic equation for ay
    def ab_fn(ax, nx, ny):
        # ax and bx
        ax = np.atleast_1d(ax)
        bx = ax + (nx * wavelength / d)

        # solve quadratic equation to get ay
        A = 2 * (ax * bx - b_dot_a) + (1 - bx ** 2) + (1 - ax ** 2)
        B = 2 * ny * wavelength / d * ((ax * bx - b_dot_a) + (1 - ax ** 2))
        C = (ax * bx - b_dot_a) ** 2 - (1 - bx ** 2) * (1 - ax ** 2) + (1 - ax ** 2) * (ny * wavelength / d) ** 2

        with np.errstate(invalid="ignore"):
            # loss of significance
            ay_pos = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
            ay_neg = (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        ay = np.stack((ay_pos, ay_neg), axis=0)
        by = ay + (ny * wavelength / d)

        # solve az, bz from unit vector equation
        with np.errstate(invalid="ignore"):
            az = -np.sqrt(1 - ax ** 2 - ay ** 2)
            bz = np.sqrt(1 - bx ** 2 - by ** 2)

        # rangle
        a = np.stack(np.broadcast_arrays(ax, ay, az), axis=-1)
        b = np.stack(np.broadcast_arrays(bx, by, bz), axis=-1)

        # since we derived our quadratic equation by squaring the equation of interest,
        # must check which solutions solve initial eqn
        soln_bad = np.abs(bx * ax + by * ay + bz * az - b_dot_a) > 1e-8

        disallowed = np.logical_or(np.any(np.isnan(a + b), axis=-1), soln_bad)
        a[disallowed] = np.nan
        b[disallowed] = np.nan

        return a, b

    def ab_from_ax(ax): return ab_fn(ax, nx, ny)
    # define function for ay from symmetry
    def ab_from_ay(ay):
        a_rev, b_rev = ab_fn(ay, ny, nx)
        a = np.array(a_rev, copy=True)
        a[..., 0] = a_rev[..., 1]
        a[..., 1] = a_rev[..., 0]

        b = np.array(b_rev, copy=True)
        b[..., 0] = b_rev[..., 1]
        b[..., 1] = b_rev[..., 0]

        return a, b

    return ab_from_ax, ab_from_ay


def solve_blazed_pattern_frequency(dx: float,
                                   dy: float,
                                   gamma: float,
                                   rot_axis: Sequence[float],
                                   wavelength: float,
                                   bf: Sequence[float],
                                   nx: int,
                                   ny: int):
    """
    Suppose we choose a desired output direction from the DMD and an order and wavelength of interest, and we
    would like to align the diffracted order b(f) with this direction and stipulate that this direction satisfies
    the blaze condition. Then determine the frequency f which allows this to be satisfied (and therefore also
    the input direction a and output direction b(0) for the main grid diffraction order)

    :param rot_axis:
    :param dx:
    :param dy:
    :param gamma:
    :param rot_axis:
    :param wavelength:
    :param bf: unit vector pointing along direction
    :param nx:
    :param ny:
    :return f, bvec, avec:
    """

    # get input vectors and output vectors for main order
    avec = solve_blaze_input(bf, gamma, rot_axis)
    bvec = solve_diffraction_output(avec, dx, dy, wavelength, nx, ny)

    f = np.stack((dx / wavelength * (bf[..., 0] - bvec[..., 0]),
                  dy / wavelength * (bf[..., 1] - bvec[..., 1])), axis=-1)

    return f, bvec, avec


def solve_diffraction_output_frq(frq: array,
                                 uvec_out: array,
                                 dx: float,
                                 dy: float,
                                 wavelength: float,
                                 order: Sequence[int]):
    """
    Suppose we want to arrange things so the output vector b(frq) points along a specific direction.
    Given that direction, solve for the required input angle and compute b(0).

    :param frq:
    :param uvec_out:
    :param dx:
    :param dy:
    :param wavelength:
    :param order: (nx, ny)
    :return b_out, uvec_in:
    """

    # given frequency, solve for DC direction
    bx_out = uvec_out[..., 0] - wavelength / dx * frq[..., 0]
    by_out = uvec_out[..., 1] - wavelength / dy * frq[..., 1]
    bz_out = np.sqrt(1 - bx_out ** 2 - by_out ** 2)
    b_out = np.stack((bx_out, by_out, bz_out), axis=-1)

    # solve for input direction
    uvec_in = solve_diffraction_input(b_out, dx, dy, wavelength, order)

    return b_out, uvec_in


# ###########################################
# 1D simulation in x-y plane and multiple wavelengths
# ###########################################
# todo: combine simulate_1d and simulate_2d
def simulate_1d(pattern: array,
                wavelengths: list,
                gamma_on: float,
                rot_axis_on: Sequence[float],
                gamma_off: float,
                rot_axis_off: Sequence[float],
                dx: float,
                dy: float,
                wx: float,
                wy: float,
                tm_ins: array,
                tm_out_offsets: Optional[array] = None,
                ndiff_orders: int = 10) -> dict:
    """
    Simulate various colors of light incident on a DMD, assuming the DMD is oriented so that the mirrors swivel in
    the same plane the incident light travels in and that this plane makes a 45-degree angle with the principle axes
    of the DMD. For more detailed discussion DMD parameters see the function simulate_dmd()

    :param pattern: binary pattern of arbitrary size
    :param wavelengths: list of wavelengths to compute
    :param gamma_on: mirror angle in ON position, relative to the DMD normal
    :param rot_axis_on:
    :param gamma_off:
    :param rot_axis_off:
    :param dx: spacing between DMD pixels in the x-direction. Same units as wavelength.
    :param dy: spacing between DMD pixels in the y-direction. Same units as wavelength.
    :param wx: width of mirrors in the x-direction. Must be < dx.
    :param wy: width of mirrors in the y-direction. Must be < dy.
    :param tm_ins: input angles in the plane of incidence
    :param tm_out_offsets: output angles relative to the angle satisfying the blaze condition
    :param ndiff_orders:
    :return data: dictionary storing simulation results
    """

    tm_ins = np.atleast_1d(tm_ins)
    ninputs = len(tm_ins)

    if tm_out_offsets is None:
        tm_out_offsets = np.linspace(-45, 45, 2400) * np.pi / 180
    tm_out_offsets = np.atleast_1d(tm_out_offsets)
    noutputs = len(tm_out_offsets)

    wavelengths = np.atleast_1d(wavelengths)
    n_wavelens = len(wavelengths)

    # input angles
    tx_ins, ty_ins = angle2xy(0, tm_ins)
    uvecs_in = xy2uvector(tx_ins, ty_ins, True)

    # blaze condition
    bvec_blaze_on = solve_blaze_output(uvecs_in, gamma_on, rot_axis_on)
    bvec_blaze_off = solve_blaze_output(uvecs_in, gamma_off, rot_axis_off)

    # variables to store simulation output data
    uvecs_out = np.zeros((ninputs, noutputs, 3))
    efields = np.zeros((ninputs, noutputs, n_wavelens), dtype=complex)
    sinc_efield_on = np.zeros(efields.shape, dtype=complex)
    sinc_efield_off = np.zeros(efields.shape, dtype=complex)

    # diffraction order predictions
    nxs = np.array(range(-ndiff_orders, ndiff_orders + 1))
    nys = -nxs
    diff_uvec_out = np.zeros((ninputs, n_wavelens, len(nxs), 3))

    # loop over input directions
    for kk in range(ninputs):
        # #########################
        # output angles track input angle
        # #########################
        _, tms_blaze_on = uvector2tmtp(*bvec_blaze_on[kk])
        tms_out = tms_blaze_on + tm_out_offsets
        txs_out, tys_out = angle2xy(np.zeros(tms_out.shape), tms_out)
        uvecs_out[kk] = xy2uvector(txs_out, tys_out, False)

        # #########################
        # do simulation
        # #########################
        for ii in range(n_wavelens):
            efields[kk, :, ii], sinc_efield_on[kk, :, ii], sinc_efield_off[kk, :, ii] \
             = simulate_dmd(pattern,
                            wavelengths[ii],
                            gamma_on,
                            gamma_off, dx, dy, wx, wy,
                            uvecs_in,
                            uvecs_out[kk])

            # get diffraction orders. Orders we want are along the antidiagonal
            diff_uvec_out[kk, ii] = solve_diffraction_output(uvecs_in[kk],
                                                             dx,
                                                             dy,
                                                             wavelengths[ii],
                                                             nxs,
                                                             nys)

    # store data
    data = {'pattern': pattern,
            'wavelengths': wavelengths,
            'gamma_on': gamma_on,
            'gamma_off': gamma_off,
            'dx': dx,
            'dy': dy,
            'wx': wx,
            'wy': wy,
            'uvecs_in': uvecs_in,
            'uvecs_out': uvecs_out,
            'uvec_out_blaze_on': bvec_blaze_on,
            'uvec_out_blaze_off': bvec_blaze_off,
            'diff_uvec_out': diff_uvec_out,
            'diff_nxs': nxs,
            'diff_nys': nys,
            'efields': efields,
            'sinc_efield_on': sinc_efield_on,
            'sinc_efield_off': sinc_efield_off}

    return data


def plot_1d_sim(data,
                colors=None,
                plot_log: bool = False,
                save_dir: Optional[str] = None,
                figsize=(18, 14)) -> (list[matplotlib.figure.Figure], list[str]):
    """
    Plot and optionally save results of simulate_1d()

    :param dict data: dictionary output from simulate_1d()
    :param list colors: list of colors, or None to use defaults
    :param bool plot_log: boolean
    :param str save_dir: directory to save data and figure results in. If None, then do not save
    :param figsize:
    :return fighs, fig_names: lists of figure handles and figure names
    """

    # ##############################
    # unpack data
    # ##############################
    pattern = data['pattern']
    wavelengths = data['wavelengths']
    n_wavelens = len(wavelengths)
    gamma_on = data['gamma_on']
    gamma_off = data['gamma_off']
    dx = data['dx']
    dy = data['dy']
    wx = data['wx']
    wy = data['wy']
    uvec_ins = data["uvecs_in"]
    uvec_outs = data["uvecs_out"]
    efields = data['efields']
    sinc_efield_on = data['sinc_efield_on']
    sinc_efield_off = data['sinc_efield_off']
    diff_uvec_out = data['diff_uvec_out']
    diff_n = data['diff_nxs']
    iz = np.where(diff_n == 0)

    # get colors if not provided
    if colors is None:
        cmap = plt.get_cmap('jet')
        colors = [cmap(ii / (n_wavelens - 1)) for ii in range(n_wavelens)]

    # decide how to scale plot
    if plot_log:
        def scale_fn(v): return np.log10(v)
    else:
        def scale_fn(v): return v

    # ##############################
    # Plot results, on different plot for each input angle
    # ##############################
    figs = []
    fig_names = []
    for kk in range(len(uvec_ins)):
        # compute useful angle data for plotting
        tx_in, ty_in = uvector2txty(uvec_ins[kk, 0], uvec_ins[kk, 1], uvec_ins[kk, 2])
        tp_in, tm_in = uvector2tmtp(uvec_ins[kk, 0], uvec_ins[kk, 1], uvec_ins[kk, 2])
        _, tms_out = uvector2tmtp(uvec_outs[kk, :, 0], uvec_outs[kk, :, 1], uvec_outs[kk, :, 2])
        _, tms_blaze_on = uvector2tmtp(*data['uvec_out_blaze_on'][kk])
        _, tms_blaze_off = uvector2tmtp(*data['uvec_out_blaze_off'][kk])

        figh = plt.figure(figsize=figsize)
        grid = figh.add_gridspec(2, 2, hspace=0.5)

        # title
        param_str = (f'spacing = {dx * 1e6:.2f}um, '
                     f'w={wx * 1e6:.2f}um, '
                     f'gamma (on,off)=({gamma_on * 180 / np.pi:.1f}, {gamma_off * 180 / np.pi:.1f}) deg\n'
                     f'theta in = ({tx_in * 180 / np.pi:.2f}, {ty_in * 180 / np.pi})deg = '
                     f'{tm_in * 180 / np.pi:.2f}deg (x-y)\n'
                     f'input unit vector = ({uvec_ins[kk, 0]:.4f}, {uvec_ins[kk, 1]:.4f}, {uvec_ins[kk, 2]:.4f})\n'
                     f'theta blaze (on,off)=({tms_blaze_on * 180 / np.pi:.2f}, {tms_blaze_off * 180 / np.pi:.2f}) '
                     f'deg in x-y dir')
        figh.suptitle(param_str)

        # ######################################
        # plot diffracted output field
        # ######################################
        ax = figh.add_subplot(grid[0, 0])

        for ii in range(n_wavelens):
            # get intensities
            intensity = np.abs(efields[kk, :, ii])**2
            intensity_sinc_on = np.abs(sinc_efield_on[kk, :, ii]) ** 2

            # normalize intensity to sinc
            im = np.argmax(np.abs(intensity))
            norm = intensity[im] / (intensity_sinc_on[im] / wx**2 / wy**2)

            # plot intensities
            ax.plot(tms_out * 180 / np.pi,
                    scale_fn(intensity / norm),
                    color=colors[ii])
            ax.plot(tms_out * 180 / np.pi,
                    scale_fn(intensity_sinc_on / (wx*wy)**2),
                    color=colors[ii],
                    ls=':')
            ax.plot(tms_out * 180 / np.pi,
                    scale_fn(np.abs(sinc_efield_off[kk, :, ii]) ** 2 / (wx*wy)**2),
                    color=colors[ii],
                    ls='--')

        ylim = ax.get_ylim()

        # plot blaze condition locations
        ax.axvline(tms_blaze_on * 180 / np.pi, color='k', ls=':')
        ax.axvline(tms_blaze_off * 180 / np.pi, color='k', ls='--')

        # plot diffraction peaks
        _, diff_tms = uvector2tmtp(diff_uvec_out[kk, ..., 0],
                                   diff_uvec_out[kk, ..., :, 1],
                                   diff_uvec_out[kk, ..., :, 2])
        for ii in range(n_wavelens):
            plt.plot(np.array([diff_tms[ii], diff_tms[ii]]) * 180 / np.pi,
                     ylim,
                     color=colors[ii],
                     ls='-')
        ax.axvline(diff_tms[0, iz] * 180 / np.pi, color='m')

        ax.set_ylim(ylim)
        ax.set_xlim([tms_blaze_on * 180 / np.pi - 7.5,
                     tms_blaze_on * 180 / np.pi + 7.5])
        ax.set_xlabel(r'$\theta_m$ (deg)')
        ax.set_ylabel('intensity (arb)')
        ax.set_title('diffraction pattern')

        # ###########################
        # plot sinc functions and wider angular range
        # ###########################
        ax = figh.add_subplot(grid[0, 1])

        for ii in range(n_wavelens):
            ax.plot(tms_out * 180 / np.pi,
                    scale_fn(np.abs(sinc_efield_on[kk, :, ii] / wx / wy)**2),
                    color=colors[ii],
                    ls=':',
                    label=f"{wavelengths[ii] * 1e9:.0f}")
            ax.plot(tms_out * 180 / np.pi,
                    scale_fn(np.abs(sinc_efield_off[kk, :, ii] / wx / wy)**2),
                    color=colors[ii],
                    ls='--')

        # plot expected blaze conditions
        ax.axvline(tms_blaze_on * 180 / np.pi, color="k", ls=":", label="blaze on")
        ax.axvline(tms_blaze_off * 180 / np.pi, color="k", ls=":", label="blaze off")

        # plot expected diffraction conditions
        for ii in range(n_wavelens):
            ax.vlines(diff_tms[ii] * 180 / np.pi, 0, 1, color=colors[ii], ls='-')

        ax.axvline(diff_tms[0, iz] * 180 / np.pi, color='m', label="0th diffraction order")

        ax.legend()
        ax.set_xlabel(r'$\theta_m$ (deg)')
        ax.set_ylabel('intensity (arb)')
        ax.set_title('blaze envelopes')

        # ###########################
        # plot pattern
        # ###########################
        ax = figh.add_subplot(grid[1, 0])
        ax.imshow(pattern, origin="lower", cmap="bone")

        ax.set_title('DMD pattern')
        ax.set_xlabel('mx')
        ax.set_ylabel('my')

        # ###########################
        # add figure to list
        # ###########################
        fname = f'dmd_sim_theta_in_{tm_in * 180 / np.pi:.03f}deg'
        fig_names.append(fname)
        figs.append(figh)

        # ###########################
        # saving
        # ###########################
        if save_dir is not None:
            figh.savefig(Path(save_dir) / f"{fname:s}.png")
            plt.close(figh)

    return figs, fig_names


# ###########################################
# 2D simulation for multiple wavelengths
# ###########################################
def simulate_2d(pattern: array,
                wavelengths: Sequence[float],
                gamma_on: float,
                rot_axis_on: Sequence[float],
                gamma_off: float,
                rot_axis_off: Sequence[float],
                dx: float,
                dy: float,
                wx: float,
                wy: float,
                tx_in: array,
                ty_in: array,
                tout_offsets: Optional[array] = None,
                ndiff_orders: int = 7) -> dict:
    """
    Simulate light incident on a DMD to determine output diffraction pattern. See simulate_dmd() for more information.

    Generally one wants to simulate many output angles but only a few input angles/wavelengths.

    :param pattern: binary pattern of arbitrary size
    :param wavelengths: list of wavelengths to compute
    :param gamma_on: mirror angle in ON position, relative to the DMD normal
    :param rot_axis_on:
    :param gamma_off:
    :param rot_axis_off:
    :param dx: spacing between DMD pixels in the x-direction. Same units as wavelength.
    :param dy: spacing between DMD pixels in the y-direction. Same units as wavelength.
    :param wx: width of mirrors in the x-direction. Must be < dx.
    :param wy: width of mirrors in the y-direction. Must be < dy.
    :param tx_in:
    :param ty_in:
    :param tout_offsets: offsets from the blaze condition to solve problem
    :param ndiff_orders:
    :return data: dictionary storing simulation results
    """

    if tout_offsets is None:
        tout_offsets = np.linspace(-25, 25, 50) * np.pi / 180
    txtx_out_offsets, tyty_out_offsets = np.meshgrid(tout_offsets, tout_offsets)

    wavelengths = np.atleast_1d(wavelengths)
    n_wavelens = len(wavelengths)

    tx_in = np.atleast_1d(tx_in)
    ty_in = np.atleast_1d(ty_in)

    # input directions
    txtx_in, tyty_in = np.meshgrid(tx_in, ty_in)
    uvecs_in = xy2uvector(txtx_in, tyty_in, True)

    # shape information
    input_shape = txtx_in.shape
    ninputs = np.prod(input_shape)
    output_shape = txtx_out_offsets.shape

    # store results
    efields = np.zeros((n_wavelens,) + input_shape + output_shape, dtype=complex)
    sinc_efield_on = np.zeros_like(efields)
    sinc_efield_off = np.zeros_like(efields)
    uvecs_out = np.zeros(input_shape + output_shape + (3,))
    # blaze condition predictions
    uvec_out_blaze_on = np.zeros(input_shape + (3,))
    uvec_out_blaze_off = np.zeros(input_shape + (3,))
    # diffraction order predictions
    diff_nx, diff_ny = np.meshgrid(range(-ndiff_orders, ndiff_orders + 1),
                                   range(-ndiff_orders, ndiff_orders + 1))
    uvec_out_diff = np.zeros((n_wavelens,) + input_shape + diff_nx.shape + (3,))

    for ii in range(ninputs):
        input_ind = np.unravel_index(ii, input_shape)

        # solve blaze condition (does not depend on wavelength)
        uvec_out_blaze_on[input_ind] = solve_blaze_output(uvecs_in[input_ind],
                                                          gamma_on,
                                                          rot_axis_on)

        uvec_out_blaze_off[input_ind] = solve_blaze_output(uvecs_in[input_ind],
                                                           gamma_off,
                                                           rot_axis_off)

        # get output directions
        tx_blaze_on, ty_blaze_on = uvector2txty(*uvec_out_blaze_on[input_ind])
        tx_outs = tx_blaze_on + txtx_out_offsets
        ty_outs = ty_blaze_on + tyty_out_offsets

        uvecs_out[input_ind] = xy2uvector(tx_outs, ty_outs, False)

        for kk in range(n_wavelens):
            # solve diffraction orders
            uvec_out_diff[kk][input_ind] = solve_diffraction_output(uvecs_in[input_ind],
                                                                    dx,
                                                                    dy,
                                                                    wavelengths[kk],
                                                                    diff_nx,
                                                                    diff_ny)

            # solve diffracted fields
            efields[kk][input_ind], sinc_efield_on[kk][input_ind], sinc_efield_off[kk][input_ind] = \
                simulate_dmd(pattern,
                             wavelengths[kk],
                             gamma_on,
                             gamma_off,
                             dx, dy, wx, wy,
                             uvecs_in[input_ind],
                             uvecs_out[input_ind])

    data = {'pattern': pattern,
            'wavelengths': wavelengths,
            'gamma_on': gamma_on,
            'gamma_off': gamma_off,
            'dx': dx,
            'dy': dy,
            'wx': wx,
            'wy': wy,
            'uvecs_in': uvecs_in,
            'uvecs_out': uvecs_out,
            'uvec_out_blaze_on': uvec_out_blaze_on,
            'uvec_out_blaze_off': uvec_out_blaze_off,
            'diff_uvec_out': uvec_out_diff,
            'diff_nxs': diff_nx,
            'diff_nys': diff_ny,
            'efields': efields,
            'sinc_efield_on': sinc_efield_on,
            'sinc_efield_off': sinc_efield_off}

    return data


def plot_2d_sim(data: dict,
                save_dir: str = 'dmd_simulation',
                figsize: Sequence[float] = (18., 14.),
                gamma: float = 0.1) -> (list[matplotlib.figure.Figure], list[str]):
    """
    Plot results from simulate_2d()

    :param dict data: dictionary object produced by simulate_2d()
    :param str save_dir:
    :param figsize:
    :param gamma:
    :return figs, fig_names:
    """

    # physical parameters
    pattern = data['pattern']
    ny, nx = pattern.shape
    wavelengths = data['wavelengths']
    dx = data['dx']
    dy = data['dy']
    wx = data['wx']
    wy = data['wy']
    gamma_on = data['gamma_on']
    gamma_off = data['gamma_off']

    # input directions
    uvecs_in = data["uvecs_in"]
    uvecs_out = data["uvecs_out"]
    uvecs_out_blaze_on = data["uvec_out_blaze_on"]
    uvecs_out_blaze_off = data["uvec_out_blaze_off"]

    # diffraction orders
    uvecs_out_diff = data["diff_uvec_out"]
    diff_nx = data['diff_nxs']
    diff_ny = data['diff_nys']
    iz = np.where(np.logical_and(diff_nx == 0, diff_ny == 0))

    # simulation results
    intensity = np.abs(data['efields'])**2
    sinc_on = np.abs(data["sinc_efield_on"])**2
    sinc_off = np.abs(data["sinc_efield_off"])**2

    # plot results
    figs = []
    fig_names = []

    input_shape = uvecs_in.shape[:-1]
    ninput = np.prod(input_shape)
    for kk in range(len(wavelengths)):
        for ii in range(ninput):
            input_ind = np.unravel_index(ii, input_shape)

            # compute all angles of interest
            tx_in, ty_in = uvector2txty(*uvecs_in[input_ind])
            tp_in, tm_in = angle2pm(tx_in, ty_in)
            tx_blaze_on, ty_blaze_on = uvector2txty(*uvecs_out_blaze_on[input_ind])
            tx_blaze_off, ty_blaze_off = uvector2txty(*uvecs_out_blaze_off[input_ind])
            diff_tx_out, diff_ty_out = uvector2txty(uvecs_out_diff[kk][input_ind][..., 0],
                                                    uvecs_out_diff[kk][input_ind][..., 1],
                                                    uvecs_out_diff[kk][input_ind][..., 2])

            param_str = (f'wavelength={wavelengths[kk] * 1e9:.0f}nm, '
                         f'dx={dx * 1e6:0.2f}um, '
                         f'w={wx * 1e6:02f}um, '
                         f'gamma (on,off)=({gamma_on * 180 / np.pi:.2f}, {gamma_off * 180 / np.pi:.2f}) deg\n'
                         f'input (tx,ty)=({tx_in * 180 / np.pi:.2f}, {ty_in * 180 / np.pi:.2f})deg '
                         f'(m,p)=({tm_in * 180 / np.pi:.2f}, {tp_in * 180/np.pi:.2f})deg\n'
                         f'input unit vector = ({uvecs_in[input_ind][0]:.4f}, '
                         f'{uvecs_in[input_ind][1]:.4f}, '
                         f'{uvecs_in[input_ind][2]:.4f})')

            tx_out, ty_out = uvector2txty(uvecs_out[input_ind][..., 0], uvecs_out[input_ind][..., 1],
                                          uvecs_out[input_ind][..., 2])
            dtout = tx_out[0, 1] - tx_out[0, 0]
            extent = [(tx_out.min() - 0.5 * dtout) * 180/np.pi,
                      (tx_out.max() + 0.5 * dtout) * 180/np.pi,
                      (ty_out.min() - 0.5 * dtout) * 180/np.pi,
                      (ty_out.max() + 0.5 * dtout) * 180/np.pi]

            # Fourier plane positions, assuming that diffraction order closest to blaze condition
            # is along the optical axis
            diff_ind = np.nanargmin(np.linalg.norm(uvecs_out_diff[kk][input_ind] - uvecs_out_blaze_on[input_ind], axis=-1))
            diff_2d_ind = np.unravel_index(diff_ind, uvecs_out_diff[kk][input_ind].shape[:-1])

            # get fourier plane positions for intensity output angles
            opt_axis = uvecs_out_diff[kk][input_ind][diff_2d_ind]
            fx, fy = uvec2dmd_frq(opt_axis, uvecs_out[input_ind], wavelengths[kk], dx, dy)
            xf, yf, _ = dmd_frq2opt_axis_uvec(fx, fy, opt_axis, opt_axis, dx, dy, wavelengths[kk])

            # get fourier plane positions for blaze conditions
            fx_blaze_on, fy_blaze_on = uvec2dmd_frq(opt_axis,
                                                    uvecs_out_blaze_on[input_ind],
                                                    wavelengths[kk],
                                                    dx, dy)
            xf_blaze_on, yf_blaze_on, _ = dmd_frq2opt_axis_uvec(fx_blaze_on,
                                                                fy_blaze_on,
                                                                opt_axis,
                                                                opt_axis,
                                                                dx, dy, wavelengths[kk])

            fx_blaze_off, fy_blaze_off = uvec2dmd_frq(opt_axis,
                                                      uvecs_out_blaze_off[input_ind],
                                                      wavelengths[kk], dx, dy)
            xf_blaze_off, yf_blaze_off, _ = dmd_frq2opt_axis_uvec(fx_blaze_off,
                                                                  fy_blaze_off,
                                                                  opt_axis,
                                                                  opt_axis,
                                                                  dx, dy, wavelengths[kk])

            # get fourier plane positions for diffraction peaks
            fx_diff, fy_diff = uvec2dmd_frq(opt_axis,
                                            uvecs_out_diff[kk][input_ind],
                                            wavelengths[kk], dx, dy)
            xf_diff, yf_diff, _ = dmd_frq2opt_axis_uvec(fx_diff,
                                                        fy_diff,
                                                        opt_axis,
                                                        opt_axis,
                                                        dx, dy, wavelengths[kk])

            fig = plt.figure(figsize=figsize)
            grid = fig.add_gridspec(2, 3)
            fig.suptitle(param_str)

            # ##################
            # intensity patterns, angular space
            # ##################
            ax = fig.add_subplot(grid[0, 0])
            ax.set_xlabel(r'$\theta_x$ outgoing (deg)')
            ax.set_ylabel(r'$\theta_y$ outgoing (deg)')
            ax.set_title('I / (wx*wy*nx*ny)**2 vs. output angle')

            ax.imshow(intensity[kk][input_ind] / (dx*dy*nx*ny)**2, extent=extent, norm=PowerNorm(gamma=gamma),
                      cmap="bone", origin="lower")
            # get xlim and ylim, we will want to keep these...
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # blaze condition
            ax.add_artist(Circle((tx_blaze_on * 180 / np.pi, ty_blaze_on * 180 / np.pi),
                          radius=1, color='r', fill=0, ls='-'))

            ax.add_artist(Circle((tx_blaze_off * 180 / np.pi, ty_blaze_off * 180 / np.pi),
                          radius=1, color='g', fill=0, ls='-'))

            # diffraction peaks
            ax.scatter(diff_tx_out * 180 / np.pi,
                       diff_ty_out * 180 / np.pi,
                       edgecolor='y',
                       facecolor='none')
            # diffraction zeroth order
            ax.scatter(diff_tx_out[iz] * 180 / np.pi,
                       diff_ty_out[iz] * 180 / np.pi,
                       edgecolor='m',
                       facecolor='none')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # ##################
            # intensity patterns, fourier plane
            # ##################
            ax = fig.add_subplot(grid[1, 0])
            ax.set_xlabel(r'$x$ (1 / lens focal len um)')
            ax.set_ylabel(r'$y$ (1 / lens focal len um)')
            ax.set_title('I / (wx*wy*nx*ny)**2 (fourier plane)')
            ax.axis("equal")

            ax.set_facecolor("k")
            ax.scatter(xf, yf, c=intensity[kk][input_ind] / (dx * dy * nx * ny) ** 2,
                       cmap="bone", norm=PowerNorm(gamma=gamma))

            # get xlim and ylim, we will want to keep these...
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # blaze condition
            ax.add_artist(Circle((xf_blaze_on, yf_blaze_on),
                                 radius=0.02, color='r', fill=0, ls='-'))

            ax.add_artist(Circle((xf_blaze_off, yf_blaze_off),
                                 radius=0.02, color='g', fill=0, ls='-'))

            # diffraction peaks
            ax.scatter(xf_diff, yf_diff, edgecolor='y', facecolor='none')
            # diffraction zeroth order
            ax.scatter(xf_diff[iz], yf_diff[iz], edgecolor='m', facecolor='none')

            # rest bounds
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # ##################
            # blaze envelopes
            # ##################
            ax = fig.add_subplot(grid[0, 1])
            ax.set_xlabel(r'$\theta_x$ outgoing')
            ax.set_ylabel(r'$\theta_y$ outgoing')
            ax.set_title('blaze condition sinc envelope (angular)')

            ax.imshow(sinc_on[kk][input_ind] / (wx*wy)**2,
                      extent=extent,
                      norm=PowerNorm(gamma=1),
                      cmap="bone",
                      origin="lower")
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # blaze condition
            ax.add_artist(Circle((tx_blaze_on * 180 / np.pi, ty_blaze_on * 180 / np.pi),
                                 radius=1, color='r', fill=0, ls='-'))

            ax.add_artist(Circle((tx_blaze_off * 180 / np.pi, ty_blaze_off * 180 / np.pi),
                                 radius=1, color='g', fill=0, ls='-'))

            # diffraction peaks
            ax.scatter(diff_tx_out * 180 / np.pi,
                       diff_ty_out * 180 / np.pi,
                       edgecolor='y',
                       facecolor='none')
            # diffraction zeroth order
            ax.scatter(diff_tx_out[iz] * 180 / np.pi,
                       diff_ty_out[iz] * 180 / np.pi,
                       edgecolor='m',
                       facecolor='none')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # ##################
            # blaze envelope, fourier plane
            # ##################
            ax = fig.add_subplot(grid[1, 1])
            ax.set_xlabel(r'$x$ (1 / lens focal len um)')
            ax.set_ylabel(r'$y$ (1 / lens focal len um)')
            ax.set_title('blaze condition sinc envelope (fourier plane)')
            ax.axis("equal")
            ax.set_facecolor("k")
            ax.scatter(xf,
                       yf,
                       c=sinc_on[kk][input_ind] / (wx*wy)**2,
                       cmap="bone",
                       norm=PowerNorm(gamma=1))
            # get xlim and ylim, we will want to keep these...
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # blaze condition
            ax.add_artist(Circle((xf_blaze_on, yf_blaze_on), radius=0.02, color='r', fill=0, ls='-'))

            ax.add_artist(Circle((xf_blaze_off, yf_blaze_off), radius=0.02, color='g', fill=0, ls='-'))

            # diffraction peaks
            ax.scatter(xf_diff, yf_diff, edgecolor='y', facecolor='none')
            # diffraction zeroth order
            ax.scatter(xf_diff[iz], yf_diff[iz], edgecolor='m', facecolor='none')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # ##################
            # DMD pattern
            # ##################
            ax = fig.add_subplot(grid[0, 2])
            ax.set_title('DMD pattern')
            ax.set_xlabel("x-position (mirrors)")
            ax.set_xlabel("y-position (mirrors)")

            ax.imshow(pattern, origin="lower", cmap="bone")

            fname = f'tx_in={tx_in:.2f}_ty_in={ty_in:.2f}_wl={wavelengths[kk] * 1e9:.0f}nm.png'
            figs.append(fig)
            fig_names.append(fname)

            # ##################
            # save results
            # ##################
            if save_dir is not None:
                save_dir = Path(save_dir)
                if not save_dir.exists():
                    save_dir.mkdir()

                fig.savefig(save_dir / fname)

    return figs, fig_names


def simulate_2d_angles(wavelengths: Sequence[float],
                       gamma: float,
                       dx: float,
                       dy: float,
                       tx_ins,
                       ty_ins,
                       ndiff_orders: int = 15) -> dict:
    """
    Determine Blaze and diffraction angles in 2D for provided input angles. For each input angle, identify the
    diffraction order which is closest to the blaze condition.

    In practice, want to use different input angles, but keep output angle fixed.
    Want to sample output angles...

    :param wavelengths: list of wavelength, in um
    :param gamma: micromirror angle in "on" position, in radians
    :param dx: x-mirror pitch, in microns
    :param dy: y-mirror pitch, in microns
    :param tx_ins: NumPy array of input angles. Output results will simulate all combinations of x- and y- input angles
    :param ty_ins: NumPy array of output angles.
    :param ndiff_orders:
    :return data: dictionary object containing simulation results
    """

    if isinstance(wavelengths, float):
        wavelengths = [wavelengths]

    n_wavelens = len(wavelengths)

    # diffraction orders to compute
    nxs, nys = np.meshgrid(range(-ndiff_orders, ndiff_orders + 1),
                           range(-ndiff_orders, ndiff_orders + 1))

    # input angles
    txtx_in, tyty_in = np.meshgrid(tx_ins, ty_ins)
    uvec_in = xy2uvector(txtx_in, tyty_in, True)

    # get output angles
    uvec_out_diff = np.zeros((n_wavelens,
                              txtx_in.shape[0],
                              txtx_in.shape[1],
                              2 * ndiff_orders + 1,
                              2 * ndiff_orders + 1, 3))
    uvecs_out_blaze = np.zeros(txtx_in.shape + (3,))
    # loop over input angles
    for ii in range(txtx_in.size):
        ind = np.unravel_index(ii, txtx_in.shape)
        uvecs_out_blaze[ind] = solve_blaze_output(uvec_in[ind], gamma)

        # loop over wavelengths
        for jj in range(n_wavelens):
            # solve diffraction orders
            uvec_out_diff[jj][ind] = solve_diffraction_output(uvec_in,
                                                              dx,
                                                              dy,
                                                              wavelengths[jj],
                                                              nxs,
                                                              nys)

    data = {'wavelengths': wavelengths,
            'gamma': gamma,
            'dx': dx,
            'dy': dy,
            'uvecs_in': uvec_in,
            'uvecs_out_blaze': uvecs_out_blaze,
            'diff_uvec_out': uvec_out_diff
            }

    return data
