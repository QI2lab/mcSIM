"""
Explore expected diffraction order strengths for difference choices of
wavelengths, pixel pitch, angle, and rotation axis
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.widgets import Slider, TextBox
import mcsim.analysis.simulate_dmd as sdmd

# diffraction orders to simulate
dx, dy = np.meshgrid(np.arange(-20, 20, dtype=int),
                     np.arange(-20, 20, dtype=int))
zero_order = np.where(np.logical_and(dx == 0, dy == 0))

cmap = "OrRd"
# ###############################
# DMD parameters
# ###############################
class DMD:
     def __init__(self, dm, wm, wavelength, rot_axis_on, gamma_on, rot_axis_off, gamma_off):
         self.dm = float(dm)
         self.wm = float(wm)
         self.wavelength = float(wavelength)
         self.gamma_on = float(gamma_on)
         self.gamma_off = float(gamma_off)
         self.rot_axis_on = np.asarray(rot_axis_on)
         self.rot_axis_off = np.asarray(rot_axis_off)


dmd = DMD(5.4,
          5,
          0.635,
          np.array([0, 0.99725, np.sqrt(1 - 0.99725**2)]),
          16.955 * np.pi/180,
          np.array([-0.99725, 0, -np.sqrt(1 - 0.99725**2)]),
          16.955 * np.pi/180,
          )

def display_2d(figsize=(8, 8)):
    """
    Create manipulatable plot to explore DMD diffraction for different input angles in several colors

    :param wavelengths: list of wavelengths (in um)
    :param gamma: DMD mirror angle in radians
    :param dx: DMD mirror pitch (in um)
    :param max_diff_order: maximum diffraction order to simulate
    :param colors: list of colors to plot various wavelengths in
    :param angle_increment: angle increment for sliders, in degrees
    :param figsize:
    :return:
    """
    # turn interactive mode off, or have problems with plot freezing
    plt.ioff()

    increment = 0.005

    # plot showing diffraction orders for each input angle, with a moveable slider
    figh = plt.figure(figsize=figsize)
    plt.suptitle("Diffraction output and blaze condition versus input angle")

    # build sliders
    axcolor = 'lightgoldenrodyellow'
    slider_height = 0.03
    slider_width = 0.65
    slider_hspace = 0.02
    slider_hstart = 0.1

    slider_axes_x = plt.axes([0.5 * (1 - slider_width),
                              slider_hstart,
                              slider_width,
                              slider_height], facecolor=axcolor)
    slider_axes_y = plt.axes([0.5 * (1 - slider_width), slider_hstart + (slider_hspace + slider_height), slider_width, slider_height],facecolor=axcolor)
    slider_axis_t1 = plt.axes([0.5 * (1 - slider_width), slider_hstart + 2*(slider_hspace + slider_height), slider_width, slider_height], facecolor=axcolor)
    slider_axis_t2 = plt.axes([0.5 * (1 - slider_width), slider_hstart + 3*(slider_hspace + slider_height), slider_width, slider_height], facecolor=axcolor)

    slider_x = Slider(slider_axes_x, 'Input direction $a_x$', -1, 1, valinit=0, valstep=increment)
    slider_y = Slider(slider_axes_y, 'Input direction $a_y$', -1, 1, valinit=0, valstep=increment)
    slider_t1 = Slider(slider_axis_t1,
                       r'$\gamma_{on}$ (deg)',
                       -20,
                       20,
                       valinit=dmd.gamma_on * 180/np.pi,
                       valstep=0.1)
    slider_t2 = Slider(slider_axis_t2,
                       r'$\gamma_{off}$ (deg)',
                       -20,
                       20,
                       valinit=dmd.gamma_off * 180/np.pi,
                       valstep=0.1)

    text_vstart = 0.025
    dm_axis = plt.axes([0.1, text_vstart, 0.1, 0.05], facecolor=axcolor)
    dm_box = TextBox(dm_axis, label="pitch ($\mu$m)", initial=str(dmd.dm))


    wlen_axis = plt.axes([0.3, text_vstart, 0.1, 0.05], facecolor=axcolor)
    wlen_box = TextBox(wlen_axis, label="$\lambda$ ($\mu$m)", initial=str(dmd.wavelength))

    # build main axis
    # [left, bottom, width, height]
    hsep = 0.05
    tics = [-1., -0.5, 0.0, 0.5, 1.]
    ax_on = plt.axes([0.05, slider_hstart + 4 * (slider_hspace + slider_height) + hsep, 0.4, 0.4])
    ax_off = plt.axes([0.55, slider_hstart + 4 * (slider_hspace + slider_height) + hsep, 0.4, 0.4])

    def update_dm(val):
        dmd.dm = float(val)
        update()

    def update_wl(val):
        dmd.wavelength = float(val)
        update()

    # function called when sliders are moved on plot
    def update(val=None):
        ax_on.clear()
        ax_off.clear()

        # get input direction from sliders
        ax_in = slider_x.val
        ay_in = slider_y.val
        az_in = -np.sqrt(1 - ax_in**2 - ay_in**2)
        a_vec = np.stack((ax_in, ay_in, az_in), axis=0)

        # define composite rotations for DMD
        dmd.gamma_on = float(slider_t1.val * np.pi/180)
        dmd.gamma_off = float(slider_t2.val * np.pi/180)

        # diffraction info
        b_all_diff = sdmd.solve_diffraction_output(a_vec,
                                                   dmd.dm,
                                                   dmd.dm,
                                                   dmd.wavelength,
                                                   dx,
                                                   dy)
        all_diff_int_on = sdmd.blaze_envelope(dmd.wavelength,
                                              dmd.gamma_on,
                                              dmd.wm,
                                              dmd.wm,
                                              b_all_diff - a_vec,
                                              dmd.rot_axis_on) ** 2
        all_diff_int_off = sdmd.blaze_envelope(dmd.wavelength,
                                               dmd.gamma_off,
                                               dmd.wm,
                                               dmd.wm,
                                               b_all_diff - a_vec,
                                               dmd.rot_axis_off) ** 2
        # blaze conditions
        b_on_out = sdmd.solve_blaze_output(a_vec, dmd.gamma_on, dmd.rot_axis_on)
        b_off_out = sdmd.solve_blaze_output(a_vec, dmd.gamma_off, dmd.rot_axis_off)


        # #####################################
        # on-mirrors plot
        # #####################################
        im = ax_on.scatter(b_all_diff[..., 0].ravel(),
                           b_all_diff[..., 1].ravel(),
                           c=all_diff_int_on.ravel(),
                           cmap=cmap,
                           norm=PowerNorm(vmin=0, vmax=1, gamma=0.2))

        # zero diffraction order
        ax_on.plot(b_all_diff[..., 0][zero_order],
                   b_all_diff[..., 1][zero_order],
                   'mx',
                   label="0th order")

        # blaze angle out
        ax_on.plot(b_on_out[:, 0],
                   b_on_out[:, 1],
                   'kx',
                   label="blaze")

        ax_on.set_xlabel('$b_x$')
        ax_on.set_ylabel('$b_y$')
        ax_on.axis("equal")
        ax_on.set_xlim([-1, 1])
        ax_on.set_ylim([-1, 1])
        ax_on.set_title("On mirrors\n"
                        f"rot axis = ({dmd.rot_axis_on[0]:.3f}, {dmd.rot_axis_on[1]:.3f}, {dmd.rot_axis_on[2]:.3f})")

        ax_on.set_xticks(tics)
        ax_on.set_yticks(tics)

        # #####################################
        # off-mirrors plot
        # #####################################
        im = ax_off.scatter(b_all_diff[..., 0].ravel(),
                            b_all_diff[..., 1].ravel(),
                            c=all_diff_int_off.ravel(),
                            cmap=cmap,
                            norm=PowerNorm(vmin=0, vmax=1, gamma=0.2))

        # zero order
        ax_off.plot(b_all_diff[..., 0][zero_order],
                   b_all_diff[..., 1][zero_order],
                   'mx',
                   label="0th order")

        # blaze angle out
        ax_off.plot(b_off_out[:, 0],
                    b_off_out[:, 1],
                    'kx',
                    label="blaze")

        ax_off.set_xlabel('$b_x$')
        ax_off.set_ylabel('$b_y$')
        ax_off.axis("equal")
        ax_off.set_xlim([-1, 1])
        ax_off.set_ylim([-1, 1])
        ax_off.set_title("Off mirrors\n"
                         f"rot axis = ({dmd.rot_axis_off[0]:.3f}, {dmd.rot_axis_off[1]:.3f}, {dmd.rot_axis_off[2]:.3f})")

        ax_off.set_xticks(tics)
        ax_off.set_yticks(tics)
        ax_off.legend()

        figh.canvas.draw_idle()

    # connect slider moves to function
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_t1.on_changed(update)
    slider_t2.on_changed(update)

    dm_box.on_submit(update_dm)
    wlen_box.on_submit(update_wl)

    # call once to ensure displays something
    update()
    # block, otherwise will freeze after a short time
    plt.show(block=True)

with np.errstate(invalid="ignore"):
    display_2d()
