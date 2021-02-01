"""
1/31/2021, Peter T. Brown
"""
import numpy as np
import matplotlib.pyplot as plt

def get_lab_coords(nx, dx, theta, gn):
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

def interp_pts(imgs, dx, dy, theta, mode="row-interp"):
    """

    :param imgs: nz x ny x nx
    :param dx: image spacing in camera space
    :param dy: distance stage moves between frames
    :param theta:
    :return:
    """
    # dy/ (dx * np.cos(theta) ) should be an integer. todo: relax this constraint
    step_size = int(dy / (dx * np.cos(theta)))

    # fix y-positions from raw images
    nx = imgs.shape[2]
    ny = imgs.shape[1]
    nz = imgs.shape[0]

    x = dx * np.arange(0, nx)
    y = dx * np.cos(theta) * np.arange(0, nx + step_size * nz)
    z = np.arange(0, ny) * np.sin(theta)

    img_unskew = np.nan * np.zeros((z.size, y.size, x.size))

    # todo: using loops for a start ... optimize later
    if mode == "row-interp": # interpolate using nearest two points on same row
        for ii in range(ny): # loop over z-positions
            for jj in range(nz): # loop over large y-position steps (moving distance btw two real frames)

                if jj < (nz - 1):
                    for kk in range(step_size): # loop over y-positions in between two frames
                        img_unskew[ii, ii + jj * step_size + kk, :] = imgs[jj, ii, :] * (step_size - kk) / step_size + \
                                                                      imgs[jj + 1, ii, :] * kk / step_size
                else:
                    img_unskew[ii, ii + jj * step_size, :] = imgs[jj, ii, :]

    elif mode == "ortho-interp": # interpolate using nearest four points
        for ii in range(ny):  # loop over z-positions
            for jj in range(nz):  # loop over large y-position steps (moving distance btw two real frames)
                if jj < (nz - 1):
                    for kk in range(step_size):  # loop over y-positions in between two frames
                        q1 = a * imgs[jj, ii, :] + b * imgs[jj, ii + 1, :]
                        qo = c * imgs[jj + 1, ii, :] + d * imgs[jj, ii + 1, :]

                        img_unskew[ii, ii + jj * step_size + kk, :] =  e * q1 + f * qo
                else:
                    img_unskew[ii, ii + jj * step_size, :] = imgs[jj, ii, :]
    else:
        raise Exception("mode must be 'row-interp' or 'ortho-interp' but was '%s'" % mode)

    return x, y, z, img_unskew

def gaussian3d_pixelated_psf(x, y, z, ds, normal, p, sf=3):
    """
    Gaussian function, accounting for image pixelation in the xy plane. This function mimics the style of the
    PSFmodels functions.

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

nx = 101
dx = 0.115
theta = 30 * np.pi / 180
# theta = 90 * np.pi / 180
normal = np.array([0, -np.sin(theta), np.cos(theta)])
# number of different pictures, separated by dz in the z-coordinate system
dy = 5 * dx * np.cos(theta)
gn = np.arange(0, 7, dy)
nz = len(gn)

# picture coordinates in coverslip frame
x, y, z = get_lab_coords(nx, dx, theta, gn)

# picture coordinates
xp = dx * np.arange(nx)
yp = xp

nc = 25
# centers = np.array([[z.mean(), y.mean(), x.mean()], [z.mean() + 0.2, y.mean() - 0.8, x.mean() + 1.5]])
centers = np.concatenate((np.random.uniform(0.25 * z.max(), 0.75 * z.max(), size=(nc, 1)),
                          np.random.uniform(0.25 * y.max(), 0.75 * y.max(), size=(nc, 1)),
                          np.random.uniform(x.min(), x.max(), size=(nc, 1))), axis=1)


imgs = np.zeros((nz, nx, nx))
for ii in range(nz):
    for jj in range(nx):
        for kk in range(nc):
            params = [1, centers[kk, 2], centers[kk, 1], centers[kk, 0], 0.3, 0.8, 0]
            imgs[ii, :, jj] += gaussian3d_pixelated_psf(x[jj], y[ii], z, dx, normal, params, sf=3)

xi, yi, zi, imgs_unskew = interp_pts(imgs, dx, dy, theta)

# plot raw data
plt.figure()
plt.suptitle("Raw data")
ncols = 6
nrows = 6
for ii in range(nz):
    extent = [xp[0] - 0.5 * dx, xp[-1] + 0.5 * dx,
              yp[-1] + 0.5 * dx, yp[0] - 0.5 * dx]

    ax = plt.subplot(nrows, ncols, ii + 1)
    ax.set_title("%0.2fum" % gn[ii])
    ax.imshow(imgs[ii], vmin=0, vmax=1, extent=extent)

    if ii == 0:
        plt.xlabel("x'")
        plt.ylabel("y'")

# plot interpolated data
plt.figure()
plt.suptitle("interpolated data")
ncols = 6
nrows = 6
for ii in range(36):
    dxi = xi[1] - xi[0]
    dyi = yi[1]- yi[0]
    extent = [xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi,
              yi[-1] + 0.5 * dyi, yi[0] - 0.5 * dyi]

    ax = plt.subplot(nrows, ncols, ii + 1)
    ax.set_title("z = %0.2fum" % zi[2 * ii])
    ax.imshow(imgs_unskew[2 * ii], vmin=0, vmax=1, extent=extent)

    if ii == 0:
        plt.xlabel("x")
        plt.ylabel("y")

# maximum intensity projection
plt.figure()
plt.imshow(np.nanmax(imgs_unskew, axis=0), vmin=0, vmax=1, extent=extent)
plt.plot(centers[:, 2], centers[:, 1], 'rx')
plt.xlabel("x (um)")
plt.ylabel("y (um)")


# plot coordinates
plt.figure()
grid = plt.GridSpec(1, 2)

ax = plt.subplot(grid[0, 0])
ax.set_title("YZ plane")
plt.plot(y, np.tile(z[None, :],[nz, 1]), 'k.')
plt.plot(centers[:, 1], centers[:, 0], 'rx')
plt.xlabel("y")
plt.ylabel("z")

ax = plt.subplot(grid[0, 1])
ax.set_title("XY plane")
for ii in range(nx):
    for jj in range(nz):
        plt.plot(y[jj], x[ii] * np.ones(y[jj].shape), 'k.')
plt.plot(centers[:,1], centers[:, 2], 'rx')
plt.xlabel("y")
plt.ylabel("x")