"""
1/31/2021, Peter T. Brown
"""
import numpy as np
import matplotlib.pyplot as plt
import camera_noise as cam

def nearest_pt_line(pt, slope, pt_line):
    """
    Get shortest distance between a point and a line.
    :param pt: (xo, yo), point of itnerest
    :param slope: slope of line
    :param pt_line: (xl, yl), point the line passes through

    :return pt: (x_near, y_near), nearest point on line
    :return d: shortest distance from point to line
    """
    xo, yo = pt
    xl, yl = pt_line
    b = yl - slope * xl

    x_int = (xo + slope * (yo - b)) / (slope**2 + 1)
    y_int = slope * x_int + b
    d = np.sqrt((xo - x_int)**2 + (yo - y_int)**2)

    return (x_int, y_int), d

def get_lab_coords(nx, dx, theta, gn):
    """
    Get laboratory coordinates (i.e. coverslip coordinates) for a stage-scanning OPM set
    :param nx:
    :param dx:
    :param theta:
    :param gn:
    :return:
    """
    x = np.arange(nx) * dx
    y = gn[:, None] + np.arange(nx)[None, :] * dx * np.cos(theta)
    z = np.arange(nx) * dx * np.sin(theta)

    return x, y, z

def cam2lab(xp, yp, gn, theta):
    x = xp
    y = gn + yp * np.cos(theta)
    z = yp * np.sin(theta)
    return x, y, z

def lab2cam(x, y, z, theta):
    """
    Get camera coordinates.
    :param x:
    :param y:
    :param z:
    :param theta:

    :return xp:
    :return yp: yp coordinate
    :return gn: distance of leading edge of camera frame from the y-axis, measured along the z-axis
    """
    xp = x
    gn = y - z / np.tan(theta)
    yp = (y - gn) / np.cos(theta)
    return xp, yp, gn

def interp_pts(imgs, dc, ds, theta, mode="row-interp"):
    """
    Interpolate OPM stage-scan data to be equally spaced in coverslip frame

    :param imgs: nz x ny x nx
    :param dc: image spacing in camera space, i.e. camera pixel size reference to object space
    :param ds: distance stage moves between frames
    :param theta:
    :return:
    """
    # ds/ (dx * np.cos(theta) ) should be an integer.
    # todo: relax this constraint if ortho-interp is used
    step_size = int(ds / (dc * np.cos(theta)))

    # fix y-positions from raw images
    nx = imgs.shape[2]
    nyp = imgs.shape[1]
    nimgs = imgs.shape[0]

    # interpolated sizes
    x = dc * np.arange(0, nx)
    y = dc * np.cos(theta) * np.arange(0, nx + step_size * nimgs)
    z = dc * np.sin(theta) * np.arange(0, nyp)
    ny = len(y)
    nz = len(z)
    # interpolated sampling spacing
    dx = dc
    dy = dc * np.cos(theta)
    dz = dc * np.sin(theta)

    img_unskew = np.nan * np.zeros((z.size, y.size, x.size))

    # todo: using loops for a start ... optimize later
    if mode == "row-interp": # interpolate using nearest two points on same row
        for ii in range(nz): # loop over z-positions
            for jj in range(nimgs): # loop over large y-position steps (moving distance btw two real frames)
                if jj < (nimgs - 1):
                    for kk in range(step_size): # loop over y-positions in between two frames
                        # interpolate to estimate value at point (y, z) = (y[ii + jj * step_size], z[ii])
                        img_unskew[ii, ii + jj * step_size + kk, :] = imgs[jj, ii, :] * (step_size - kk) / step_size + \
                                                                      imgs[jj + 1, ii, :] * kk / step_size
                else:
                    img_unskew[ii, ii + jj * step_size, :] = imgs[jj, ii, :]

    # todo: this mode can be generalized to not use dy a multiple of dx
    elif mode == "ortho-interp": # interpolate using nearest four points.
        for ii in range(nz):  # loop over z-positions
            for jj in range(nimgs):  # loop over large y-position steps (moving distance btw two real frames)
                if jj < (nimgs - 1):
                    for kk in range(step_size):  # loop over y-positions in between two frames
                        # interpolate to estimate value at point (y, z) = (y[ii + jj * step_size + kk], z[ii])
                        pt_now = (y[ii + jj * step_size + kk], z[ii])

                        # find nearest point on line passing through (y[jj * step_size], 0)
                        pt_n1, dist_1 = nearest_pt_line(pt_now, np.tan(theta), (y[jj * step_size], 0))
                        dist_along_line1 = np.sqrt( (pt_n1[0] - y[jj * step_size])**2 + pt_n1[1]**2) / dc
                        # as usual, need to round to avoid finite precision floor/ceiling issues if number is already an integer
                        i1_low = int(np.floor(np.round(dist_along_line1, 14)))
                        i1_high = int(np.ceil(np.round(dist_along_line1, 14)))

                        if np.round(dist_1, 14) == 0:
                            q1 = imgs[jj, i1_low, :]
                        elif i1_low < 0 or i1_high >= nyp:
                            q1 = np.nan
                        else:
                            d1 = dist_along_line1 - i1_low
                            q1 = (1 - d1) * imgs[jj, i1_low, :] + d1 * imgs[jj, i1_high, :]

                        # find nearest point on line passing through (y[(jj + 1) * step_size], 0)
                        pt_no, dist_o = nearest_pt_line(pt_now, np.tan(theta), (y[(jj + 1) * step_size], 0))
                        dist_along_line0 = np.sqrt((pt_no[0] - y[(jj + 1) * step_size]) ** 2 + pt_no[1] ** 2) / dc
                        io_low = int(np.floor(np.round(dist_along_line0, 14)))
                        io_high = int(np.ceil(np.round(dist_along_line0, 14)))

                        if np.round(dist_o, 14) == 0:
                            qo = imgs[jj + 1, i1_low, :]
                        elif io_low < 0 or io_high >= nyp:
                            qo = np.nan
                        else:
                            do = dist_along_line0 - io_low
                            qo = (1 - do) * imgs[jj + 1, io_low, :] + do * imgs[jj + 1, io_high, :]

                        # weighted average of qo and q1 based on their distance
                        img_unskew[ii, ii + jj * step_size + kk, :] = (q1 * dist_o + qo * dist_1) / (dist_o + dist_1)
                else:
                    img_unskew[ii, ii + jj * step_size, :] = imgs[jj, ii, :]
    else:
        raise Exception("mode must be 'row-interp' or 'ortho-interp' but was '%s'" % mode)

    return x, y, z, img_unskew

def gaussian3d_pixelated_psf(x, y, z, ds, normal, p, sf=3):
    """
    Gaussian function, accounting for image pixelation in the xy plane. This function mimics the style of the
    PSFmodels functions.

    todo: vectorize appropriately

    :param dx: pixel size in um
    :param nx: number of pixels (must be odd)
    :param z: coordinates of z-planes to evaluate function at
    :param p: [A, cx, cy, cz, sxy, sz, bg]
    :param wavelength: in um
    :param ni: refractive index
    :param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
    points in the pixel.
    :return:
    """

    # generate new points in pixel
    pts = ds * (np.arange(1 / sf / 2, 1 - 1 / sf /2, 1 / sf) - 0.5)
    xp, yp = np.meshgrid(pts, pts)
    zp = np.zeros(xp.shape)

    # rotate points to correct position using normal vector
    # for now we will fix x, but lose generality
    eyp = np.cross(normal, np.array([1, 0, 0]))
    mat_r2rp = np.concatenate((np.array([1, 0, 0])[:, None], eyp[:, None], normal[:, None]), axis=1)
    result = mat_r2rp.dot(np.concatenate((xp.ravel()[None, :], yp.ravel()[None, :], zp.ravel()[None, :]), axis=0))
    xs, ys, zs = result

    # now must add these to each point x, y, z
    xx_s = x[..., None] + xs[None, ...]
    yy_s = y[..., None] + ys[None, ...]
    zz_s = z[..., None] + zs[None, ...]

    psf_s = np.exp( - (xx_s - p[1])**2 / 2 / p[4]**2 - (yy_s - p[2])**2 / 2/ p[4]**2 - (zz_s - p[3])**2 / 2/ p[5]**2)
    norm = np.sum(np.exp(-xs**2 / 2 / p[4]**2 - ys**2 / 2 / p[5]**2))

    psf = p[0] / norm * np.sum(psf_s, axis=-1) + p[-1]

    return psf

if __name__ == "__main__":
    nx = 101
    dx = 0.115 # pixel size
    theta = 30 * np.pi / 180
    # theta = 90 * np.pi / 180
    normal = np.array([0, -np.sin(theta), np.cos(theta)])
    # number of different pictures, separated by dz in the z-coordinate system
    dy = 2 * dx * np.cos(theta)
    dz = dx * np.sin(theta)
    gn = np.arange(0, 15, dy)
    nz = len(gn)

    # picture coordinates in coverslip frame
    x, y, z = get_lab_coords(nx, dx, theta, gn)

    # picture coordinates
    xp = dx * np.arange(nx)
    yp = xp

    nc = 10
    # centers = np.array([[z.mean(), y.mean(), x.mean()], [z.mean() + 0.2, y.mean() - 0.8, x.mean() + 1.5]])
    centers = np.concatenate((np.random.uniform(0.25 * z.max(), 0.75 * z.max(), size=(nc, 1)),
                              np.random.uniform(0.25 * y.max(), 0.75 * y.max(), size=(nc, 1)),
                              np.random.uniform(x.min(), x.max(), size=(nc, 1))), axis=1)


    imgs_opm = np.zeros((nz, nx, nx))
    for ii in range(nz):
        for jj in range(nx):
            for kk in range(nc):
                params = [1, centers[kk, 2], centers[kk, 1], centers[kk, 0], 0.3, 1.2, 0]
                imgs_opm[ii, :, jj] += gaussian3d_pixelated_psf(x[jj], y[ii], z, dx, normal, params, sf=3)

    # add shot-noise and gaussian readout noise
    nphotons = 100
    bg = 100
    gain = 2
    noise = 5
    imgs_opm, _, _ = cam.simulated_img(imgs_opm, nphotons, gain, bg, noise, use_otf=False)

    # interpolate images
    xi, yi, zi, imgs_unskew = interp_pts(imgs_opm, dx, dy, theta, mode="row-interp")
    _, _, _, imgs_unskew2 = interp_pts(imgs_opm, dx, dy, theta, mode="ortho-interp")
    dxi = xi[1] - xi[0]
    dyi = yi[1] - yi[0]
    dzi = zi[1]- zi[0]

    # get gt in interpolated space
    imgs_square = np.zeros((len(zi), len(yi), len(xi)))
    for ii in range(len(zi)):
        for jj in range(len(xi)):
            for kk in range(nc):
                params = [1, centers[kk, 2], centers[kk, 1], centers[kk, 0], 0.3, 1.2, 0]
                imgs_square[ii, :, jj] += gaussian3d_pixelated_psf(xi[jj], yi, zi[ii], dxi, np.array([0, 0, 1]), params, sf=3)
    # add noise
    imgs_square, _, _ = cam.simulated_img(imgs_square, nphotons, gain, bg, noise, use_otf=False)
    # nan-mask region outside what we get from the OPM
    imgs_square[np.isnan(imgs_unskew)] = np.nan

    # plot raw data
    plt.figure()
    plt.suptitle("Raw data")
    ncols = int(np.ceil(np.sqrt(nz)) + 1)
    nrows = int(np.ceil(nz / ncols))
    for ii in range(nz):
        extent = [xp[0] - 0.5 * dx, xp[-1] + 0.5 * dx,
                  yp[-1] + 0.5 * dx, yp[0] - 0.5 * dx]

        ax = plt.subplot(nrows, ncols, ii + 1)
        ax.set_title("%0.2fum" % gn[ii])
        ax.imshow(imgs_opm[ii], vmin=bg, vmax=bg + gain * nphotons, extent=extent)

        if ii == 0:
            plt.xlabel("x'")
            plt.ylabel("y'")

    # plot interpolated data
    plt.figure()
    plt.suptitle("interpolated data, row interp")
    ncols = 6
    nrows = 6
    for ii in range(36):
        extent = [xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi,
                  yi[-1] + 0.5 * dyi, yi[0] - 0.5 * dyi]

        ax = plt.subplot(nrows, ncols, ii + 1)
        ax.set_title("z = %0.2fum" % zi[2 * ii])
        ax.imshow(imgs_unskew[2 * ii], vmin=bg, vmax=bg + gain * nphotons, extent=extent)

        if ii == 0:
            plt.xlabel("x")
            plt.ylabel("y")

    # plot interpolated data, using other scheme
    plt.figure()
    plt.suptitle("interpolated data, ortho-interp")
    ncols = 6
    nrows = 6
    for ii in range(36):
        extent = [xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi,
                  yi[-1] + 0.5 * dyi, yi[0] - 0.5 * dyi]

        ax = plt.subplot(nrows, ncols, ii + 1)
        ax.set_title("z = %0.2fum" % zi[2 * ii])
        ax.imshow(imgs_unskew2[2 * ii], vmin=bg, vmax=bg + gain * nphotons, extent=extent)

        if ii == 0:
            plt.xlabel("x")
            plt.ylabel("y")

    # maximum intensity projection
    plt.figure()
    plt.suptitle("Max int projection, row interp")

    plt.subplot(1, 3, 1)
    plt.imshow(np.nanmax(imgs_unskew, axis=0).transpose(), vmin=bg, vmax=bg + gain * nphotons,
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, xi[-1] + 0.5 * dxi, xi[0] - 0.5 * dxi])
    plt.plot(centers[:, 1], centers[:, 2], 'rx')
    plt.xlabel("Y (um)")
    plt.ylabel("X (um)")
    plt.title("XY")

    plt.subplot(1, 3, 2)
    plt.imshow(np.nanmax(imgs_unskew, axis=1), vmin=bg, vmax=bg + gain * nphotons,
               extent=[xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi, zi[-1] + 0.5 * dzi, zi[0] - 0.5 * dzi])
    plt.plot(centers[:, 2], centers[:, 0], 'rx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")
    plt.title("XZ")

    plt.subplot(1, 3, 3)
    plt.imshow(np.nanmax(imgs_unskew, axis=2), vmin=bg, vmax=bg + gain * nphotons,
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, zi[-1] + 0.5 * dzi, zi[0] - 0.5 * dzi])
    plt.plot(centers[:, 1], centers[:, 0], 'rx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")
    plt.title("YZ")

    # ortho interp
    plt.figure()
    plt.suptitle("Max int projection, ortho interp")

    plt.subplot(1, 3, 1)
    plt.imshow(np.nanmax(imgs_unskew2, axis=0).transpose(), vmin=bg, vmax=bg + gain * nphotons,
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, xi[-1] + 0.5 * dxi, xi[0] - 0.5 * dxi])
    plt.plot(centers[:, 1], centers[:, 2], 'rx')
    plt.xlabel("Y (um)")
    plt.ylabel("X (um)")
    plt.title("XY")

    plt.subplot(1, 3, 2)
    plt.imshow(np.nanmax(imgs_unskew2, axis=1), vmin=bg, vmax=bg + gain * nphotons,
               extent=[xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi, zi[-1] + 0.5 * dzi, zi[0] - 0.5 * dzi])
    plt.plot(centers[:, 2], centers[:, 0], 'rx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")
    plt.title("XZ")

    plt.subplot(1, 3, 3)
    plt.imshow(np.nanmax(imgs_unskew2, axis=2), vmin=bg, vmax=bg + gain * nphotons,
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, zi[-1] + 0.5 * dzi, zi[0] - 0.5 * dzi])
    plt.plot(centers[:, 1], centers[:, 0], 'rx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")
    plt.title("YZ")

    # ground truth in these coords
    plt.figure()
    plt.suptitle("gt")

    plt.subplot(1, 3, 1)
    plt.imshow(np.nanmax(imgs_square, axis=0).transpose(), vmin=bg, vmax=bg + gain * nphotons,
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, xi[-1] + 0.5 * dxi, xi[0] - 0.5 * dxi])
    plt.plot(centers[:, 1], centers[:, 2], 'rx')
    plt.xlabel("Y (um)")
    plt.ylabel("X (um)")
    plt.title("XY")

    plt.subplot(1, 3, 2)
    plt.imshow(np.nanmax(imgs_square, axis=1), vmin=bg, vmax=bg + gain * nphotons,
               extent=[xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi, zi[-1] + 0.5 * dzi, zi[0] - 0.5 * dzi])
    plt.plot(centers[:, 2], centers[:, 0], 'rx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")
    plt.title("XZ")

    plt.subplot(1, 3, 3)
    plt.imshow(np.nanmax(imgs_square, axis=2), vmin=bg, vmax=bg + gain * nphotons,
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, zi[-1] + 0.5 * dzi, zi[0] - 0.5 * dzi])
    plt.plot(centers[:, 1], centers[:, 0], 'rx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")
    plt.title("YZ")

    # plot coordinates
    plt.figure()
    grid = plt.GridSpec(1, 2)
    plt.suptitle("Coordinates, dy=%0.2fum, dx=%0.2fum, dyi=%0.2f, dzi=%0.2f, theta=%0.2fdeg" % (dy, dx, dyi, dzi, theta * 180/np.pi))

    ax = plt.subplot(grid[0, 0])
    ax.set_title("YZ plane")
    yiyi, zizi = np.meshgrid(yi, zi)
    plt.plot(yiyi, zizi, 'bx')
    plt.plot(y, np.tile(z[None, :],[nz, 1]), 'k.')
    plt.plot(centers[:, 1], centers[:, 0], 'rx')
    plt.xlabel("y")
    plt.ylabel("z")
    plt.axis('equal')

    ax = plt.subplot(grid[0, 1])
    ax.set_title("XY plane")
    yiyi, xixi = np.meshgrid(yi, xi)
    plt.plot(yiyi, xixi, 'bx')
    for ii in range(nx):
        for jj in range(nz):
            plt.plot(y[jj], x[ii] * np.ones(y[jj].shape), 'k.')
    plt.plot(centers[:,1], centers[:, 2], 'rx')
    plt.xlabel("y")
    plt.ylabel("x")
    plt.axis("equal")