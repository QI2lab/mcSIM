"""
Tools for fitting data using non-linear least squares, and other approaches.
"""
import copy
import numpy as np
import scipy
import scipy.optimize


def fit_model(img, model_fn, init_params, fixed_params=None, sd=None, bounds=None, model_jacobian=None, **kwargs):
    """
    Fit 2D model function to an image. Any Nan values in the image will be ignored. Wrapper for fit_least_squares

    :param np.array img: nd array
    :param model_fn: function f(p)
    :param list[float] init_params: p = [p1, p2, ..., pn]
    :param list[boolean] fixed_params: list of boolean values, same size as init_params. If None, no parameters will be fixed.
    :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
    deviation of the mean. If None, then will use a value of 1 for all points. As long as these values are all the same
    they will not affect the optimization results, although they will affect chi squared.
    :param tuple[tuple[float]] bounds: (lbs, ubs). If None, -/+ infinity used for all parameters.
    :param model_jacobian: Jacobian of the model function as a list, [df/dp[0], df/dp[1], ...]. If None, no jacobian used.
    :param kwargs: additional key word arguments will be passed through to scipy.optimize.least_squares
    :return:
    """
    to_use = np.logical_not(np.isnan(img))

    # if all sd's are nan or zero, set to 1
    if sd is None or np.all(np.isnan(sd)) or np.all(sd == 0):
        sd = np.ones(img.shape)

    # handle uncertainties that will cause fitting to fail
    if np.any(sd == 0) or np.any(np.isnan(sd)):
        sd[sd == 0] = np.nanmean(sd[sd != 0])
        sd[np.isnan(sd)] = np.nanmean(sd[sd != 0])

    # function to be optimized
    def err_fn(p): return np.divide(model_fn(p)[to_use].ravel() - img[to_use].ravel(), sd[to_use].ravel())

    # if it was passed, use model jacobian
    if model_jacobian is not None:
        def jac_fn(p): return [v[to_use] / sd[to_use] for v in model_jacobian(p)]
    else:
        jac_fn = None

    results = fit_least_squares(err_fn, init_params, fixed_params=fixed_params, bounds=bounds,
                                model_jacobian=jac_fn, **kwargs)

    return results


def fit_least_squares(model_fn, init_params, fixed_params=None, bounds=None, model_jacobian=None, **kwargs):
    """
    Wrapper for non-linear least squares fi t function scipy.optimize.least_squares which handles fixing parameters.

    :param model_fn: function f(p)
    :param list[float] init_params: p = [p1, p2, ..., pn]
    :param list[boolean] fixed_params: list of boolean values, same size as init_params. If None, no parameters will be fixed.
    :param tuple[tuple[float]] bounds: (lbs, ubs). If None, -/+ infinity used for all parameters.
    :param model_jacobian: Jacobian of the model function as a list, [df/dp[0], df/dp[1], ...]. If None, no jacobian used.
    :param kwargs: additional key word arguments will be passed through to scipy.optimize.least_squares
    :return:
    """

    # get default fixed parameters
    if fixed_params is None:
        fixed_params = [False for _ in init_params]

    # default bounds
    if bounds is None:
        bounds = (tuple([-np.inf] * len(init_params)), tuple([np.inf] * len(init_params)))

    init_params = np.array(init_params, copy=True)
    # ensure initial parameters within bounds, if not fixed
    for ii in range(len(init_params)):
        if (init_params[ii] < bounds[0][ii] or init_params[ii] > bounds[1][ii]) and not fixed_params[ii]:
            raise ValueError("Initial parameter at index %d had value %0.2g, which was outside of bounds (%0.2g, %0.2g"
                             % (ii, init_params[ii], bounds[0][ii], bounds[1][ii]))

    if np.any(np.isnan(init_params)):
        raise ValueError("init_params cannot include nans")

    if np.any(np.isnan(bounds)):
        raise ValueError("bounds cannot include nans")

    # if some parameters are fixed, we need to hide them from the fit function to produce correct covariance, etc.
    # Idea: map the "reduced" (i.e. not fixed) parameters onto the full parameter list.
    # do this by looking at each parameter. If it is supposed to be "fixed" substitute the initial parameter. If not,
    # then get the next value from pfree. We find the right index of pfree by summing the number of previously unfixed parameters
    free_inds = [int(np.sum(np.logical_not(fixed_params[:ii]))) for ii in range(len(fixed_params))]

    def pfree2pfull(pfree):
        return np.array([pfree[free_inds[ii]] if not fp else init_params[ii] for ii, fp in enumerate(fixed_params)])

    # map full parameters to reduced set
    def pfull2pfree(pfull): return np.array([p for p, fp in zip(pfull, fixed_params) if not fp])

    # function to minimize the sum of squares of, now as a function of only the free parameters
    def err_fn_pfree(pfree): return model_fn(pfree2pfull(pfree))

    if model_jacobian is not None:
        def jac_fn_free(pfree): return pfull2pfree(model_jacobian(pfree2pfull(pfree))).transpose()
    init_params_free = pfull2pfree(init_params)
    bounds_free = (tuple(pfull2pfree(bounds[0])), tuple(pfull2pfree(bounds[1])))

    # non-linear least squares fit
    if model_jacobian is None:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free, **kwargs)
    else:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free,
                                                jac=jac_fn_free, x_scale='jac', **kwargs)
    pfit = pfree2pfull(fit_info['x'])

    # calculate chi squared
    nfree_params = np.sum(np.logical_not(fixed_params))
    red_chi_sq = np.sum(np.square(model_fn(pfit))) / (model_fn(init_params).size - nfree_params)

    # calculate covariances
    try:
        jacobian = fit_info['jac']
        cov_free = red_chi_sq * np.linalg.inv(jacobian.transpose().dot(jacobian))
    except np.linalg.LinAlgError:
        cov_free = np.nan * np.zeros((jacobian.shape[1], jacobian.shape[1]))

    cov = np.nan * np.zeros((len(init_params), len(init_params)))
    ii_free = 0
    for ii, fpi in enumerate(fixed_params):
        jj_free = 0
        for jj, fpj in enumerate(fixed_params):
            if not fpi and not fpj:
                cov[ii, jj] = cov_free[ii_free, jj_free]
                jj_free += 1
                if jj_free == nfree_params:
                    ii_free += 1

    result = {'fit_params': pfit, 'chi_squared': red_chi_sq, 'covariance': cov,
              'init_params': init_params, 'fixed_params': fixed_params, 'bounds': bounds,
              'cost': fit_info['cost'], 'optimality': fit_info['optimality'],
              'nfev': fit_info['nfev'], 'njev': fit_info['njev'], 'status': fit_info['status'],
              'success': fit_info['success'], 'message': fit_info['message']}

    return result


def get_moments(img, order=1, coords=None, dims=None):
    """
    Calculate moments of distribution of arbitrary size
    :param img:
    :param order: order of moments to be calculated
    :param coords: list of coordinate arrays for each dimension e.g. [y, x], where y, x etc. are 1D arrays
    :param dims: dimensions to be summed over. For example, given roi_size 3D array of size Nz x Ny x Nz, calculate the 2D
    moments of each slice by setting dims = [1, 2]
    :return:
    """
    if dims is None:
        dims = range(img.ndim)

    if coords is None:
        coords = [np.arange(s) for ii, s in enumerate(img.shape) if ii in dims]
    # ensure coords are float arrays to avoid overflow issues
    coords = [np.array(c, dtype=np.float) for c in coords]

    if len(dims) != len(coords):
        raise ValueError('dims and coordinates must have the same length')

    # weight summing only over certain dimensions
    w = np.nansum(img, axis=tuple(dims), dtype=np.float)

    # as trick to avoid having to meshgrid any of the coordinates, we can use NumPy's array broadcasting. Because this
    # looks at the trailing array dimensions, we need to swap our desired axis to be the last dimension, multiply by the
    # coordinates to do the broadcasting, and then swap back
    moments = [np.nansum(np.swapaxes(np.swapaxes(img, ii, img.ndim-1) * c**order, ii, img.ndim-1),
               axis=tuple(dims), dtype=np.float) / w
               for ii, c in zip(dims, coords)]

    return moments


# fit data to gaussians
def fit_gauss1d(y, init_params=None, fixed_params=None, sd=None, x=None, bounds=None):
    """
    Fit 1D Gaussian

    :param y:
    :param init_params: [A, cx, sx, bg]
    :param fixed_params:
    :param sd:
    :param x:
    :param bounds:
    :return:
    """

    # get coordinates if not provided
    if x is None:
        x = np.arange(len(y))

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 4
    else:
        init_params = copy.deepcopy(init_params)

    # guess reasonable parameters if not provided
    if np.any([ip is None for ip in init_params]):
        to_use = np.logical_not(np.isnan(y))

        bg = np.nanmean(y.ravel())
        A = np.max(y[to_use].ravel()) - bg

        cx, = get_moments(y, order=1, coords=[x])
        m2x, = get_moments(y, order=2, coords=[x])
        sx = np.sqrt(m2x - cx ** 2)

        ip_default = [A, cx, sx, bg]

        # set any parameters that were None to the default values
        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    if bounds is None:
        bounds = ((-np.inf, x.min(), 0, -np.inf),
                  (np.inf, x.max(), x.max() - x.min(), np.inf))

    fn = lambda p: gauss_fn(x, np.zeros(x.shape), [p[0], p[1], 0, p[2], 1, p[3], 0])
    jacob_fn = lambda p: gauss_jacobian(x, np.zeros(x.shape), [p[0], p[1], 0, p[2], 1, p[3], 0])

    result = fit_model(y, fn, init_params, fixed_params=fixed_params,
                       sd=sd, bounds=bounds, model_jacobian=jacob_fn)

    pfit = result['fit_params']
    fit_fn = lambda x: gauss_fn(x, np.zeros(x.shape), [pfit[0], pfit[1], 0, pfit[2], 1, pfit[3], 0])

    return result, fit_fn


def fit_gauss2d(img, init_params=None, fixed_params=None, sd=None, xx=None, yy=None, bounds=None):
    """
    Fit 2D gaussian function. The angle theta is defined clockwise from the x- (or y-) axis. NOTE: be careful
    with this when looking at results using e.g. matplotlib.imshow, as this will display the negative y-axis on top.

    :param img: 2D image to fit
    :param init_params: [A, cx, cy, sx, sy, bg, theta]
    :param fixed_params: list of boolean values, same size as init_params.
    :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
    deviation of the mean
    :param xx: 2D array, same size as image (use this instead of 1D array because want to preserve ability to fit on
    non-regularly spaced grids, etc.)
    :param yy:
    :param bounds: (lbs, ubs)
    :return dict results:
    :return fit_fn:
    """

    # get coordinates if not provided
    if xx is None or yy is None:
        xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 7
    else:
        init_params = copy.deepcopy(init_params)

    # guess reasonable parameters if not provided
    if np.any([ip is None for ip in init_params]):
        to_use = np.logical_not(np.isnan(img))

        bg = np.nanmean(img.ravel())
        A = np.max(img[to_use].ravel()) - bg

        cy, cx = get_moments(img, order=1, coords=[yy[:, 0], xx[0, :]])
        m2y, m2x = get_moments(img, order=2, coords=[yy[:, 0], xx[0, :]])
        with np.errstate(invalid='ignore'):
            sx = np.sqrt(m2x - cx ** 2)
            sy = np.sqrt(m2y - cy ** 2)

        ip_default = [A, cx, cy, sx, sy, bg, 0]

        # set any parameters that were None to the default values
        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    # replace any bounds which are none with default guesses
    lbs_default = (-np.inf, xx.min(), yy.min(), 0, 0, -np.inf, -np.inf)
    ubs_default = (np.inf, xx.max(), yy.max(), xx.max() - xx.min(), yy.max() - yy.min(), np.inf, np.inf)

    if bounds is None:
        bounds = (lbs_default, ubs_default)
    else:
        lbs = tuple([b if b is not None else lbs_default[ii] for ii, b in enumerate(bounds[0])])
        ubs = tuple([b if b is not None else ubs_default[ii] for ii, b in enumerate(bounds[1])])
        bounds = (lbs, ubs)

    # do fitting
    result = fit_model(img, lambda p: gauss_fn(xx, yy, p), init_params, fixed_params=fixed_params,
                       sd=sd, bounds=bounds, model_jacobian=lambda p: gauss_jacobian(xx, yy, p))

    # model function
    def fit_fn(x, y): return gauss_fn(x, y, result['fit_params'])

    return result, fit_fn


def fit_sum_gauss2d(img, ngaussians, init_params, fixed_params=None, sd=None, xx=None, yy=None, bounds=None):
    """
    Fit 2D gaussian function. The angle theta is defined clockwise from the x- (or y-) axis. NOTE: be careful
    with this when looking at results using e.g. matplotlib.imshow, as this will display the negative y-axis on top.

    :param img: 2D image to fit
    :param init_params: [A1, cx1, cy1, sx1, sy1, theta1, A2, cx2, ..., thetan, bg]
    :param fixed_params: list of boolean values, same size as init_params.
    :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
    deviation of the mean
    :param xx: 2D array, same size as image (use this instead of 1D array because want to preserve ability to fit on
    non-regularly spaced grids, etc.)
    :param yy:
    :param bounds: (lbs, ubs)
    :return:
    """

    # get coordinates if not provided
    if xx is None or yy is None:
        xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    nparams = 6 * ngaussians + 1
    # get default initial parameters
    if init_params is None:
        init_params = [None] * nparams
    else:
        init_params = copy.deepcopy(init_params)

    if bounds is None:
        bounds = [[-np.inf, xx.min(), yy.min(), 0, 0, -np.inf] * ngaussians + [-np.inf],
                  [ np.inf, xx.max(), yy.max(), xx.max() - xx.min(), yy.max() - yy.min(), np.inf] * ngaussians + [np.inf]]

    result = fit_model(img, lambda p: sum_gauss_fn(xx, yy, p), init_params, fixed_params=fixed_params,
                       sd=sd, bounds=bounds, model_jacobian=lambda p: sum_gauss_jacobian(xx, yy, p))

    pfit = result['fit_params']

    def fn(x, y):
        return sum_gauss_fn(x, y, pfit)

    return result, fn


def fit_half_gauss1d(y, init_params=None, fixed_params=None, sd=None, x=None, bounds=None):
    """
    Fit function that has two Gaussian halves with different sigmas and offsets but match smoothly at cx

    :param y:
    :param init_params: [A1, cx, sx1, bg1, sx2, bg2]
    :param fixed_params:
    :param sd:
    :param x:
    :param bounds:
    :return:
    """

    # get coordinates if not provided
    if x is None:
        x = np.arange(len(y))

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 6
    else:
        # init_params = copy.deepcopy(init_params)
        init_params = [p for p in init_params]

    # guess reasonable parameters if not provided
    if np.any([ip is None for ip in init_params]):
        to_use = np.logical_not(np.isnan(y))

        bg = np.nanmean(y.ravel())
        A = np.max(y[to_use].ravel()) - bg

        cx, = get_moments(y, order=1, coords=[x])
        m2x, = get_moments(y, order=2, coords=[x])
        sx = np.sqrt(m2x - cx ** 2)

        ip_default = [A, cx, sx, bg, sx, bg]

        # set any parameters that were None to the default values
        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    if bounds is None:
        bounds = ((-np.inf, x.min(), 0, -np.inf, 0, -np.inf),
                  (np.inf, x.max(), x.max() - x.min(), np.inf, x.max() - x.min(), np.inf))

    hg_fn = lambda x, p: (p[0] * np.exp(-(x - p[1])**2 / (2*p[2]**2)) + p[3]) * (x < p[1]) + \
                         ((p[0] + p[3] - p[5]) * np.exp(-(x - p[1])**2 / (2*p[4]**2)) + p[5]) * (x >= p[1])

    result = fit_model(y, lambda p: hg_fn(x, p), init_params, fixed_params=fixed_params, sd=sd, bounds=bounds)

    pfit = result['fit_params']
    fit_fn = lambda x: hg_fn(x, pfit)

    return result, fit_fn


# gaussians and jacobians
def gauss_fn(x, y, p):
    """
    Rotated 2D gaussian function. The angle theta is defined clockwise from the x- (or y-) axis. NOTE: be careful
    with this when looking at results using e.g. matplotlib.imshow, as this will display the negative y-axis on top.

    :param x: x-coordinates to evaluate function at.
    :param y: y-coordinates to evaluate function at. Either same size as x, or broadcastable with x.
    :param p: [A, cx, cy, sxrot, syrot, bg, theta]
    :return value:
    """
    if len(p) != 7:
        raise ValueError("parameter list p must have length 7")

    xrot = np.cos(p[6]) * (x - p[1]) - np.sin(p[6]) * (y - p[2])
    yrot = np.cos(p[6]) * (y - p[2]) + np.sin(p[6]) * (x - p[1])
    return p[0] * np.exp(-xrot ** 2 / (2 * p[3] ** 2) - yrot ** 2 / (2 * p[4] ** 2)) + p[5]


def gauss_jacobian(x, y, p):
    """
    Jacobian of gauss_fn

    :param x:
    :param y:
    :param p: [A, cx, cy, sx, sy, bg, theta]
    :return value:
    """
    if len(p) != 7:
        raise ValueError("parameter list p must have length 7")

    # useful functions that show up in derivatives
    xrot = np.cos(p[6]) * (x - p[1]) - np.sin(p[6]) * (y - p[2])
    yrot = np.cos(p[6]) * (y - p[2]) + np.sin(p[6]) * (x - p[1])
    exps = np.exp(-xrot**2 / (2 * p[3] ** 2) -yrot**2 / (2 * p[4] ** 2))

    bcast_shape = (x + y).shape

    return [exps,
            p[0] * exps * (xrot / p[3]**2 * np.cos(p[6]) + yrot / p[4]**2 * np.sin(p[6])),
            p[0] * exps * (yrot / p[4]**2 * np.cos(p[6]) - xrot / p[3]**2 * np.sin(p[6])),
            p[0] * exps * xrot ** 2 / p[3] ** 3,
            p[0] * exps * yrot ** 2 / p[4] ** 3,
            np.ones(bcast_shape),
            p[0] * exps * xrot * yrot * (1 / p[3]**2 - 1 / p[4]**2)]


def sum_gauss_fn(x, y, p):
    """
    Sum of n 2D gaussians
    :param x:
    :param y:
    :param p: [A1, cx1, cx2, sx1, sx2, theta1, A2, ..., thetan, bg]
    :return:
    """
    if len(p) % 6 != 1:
        raise ValueError("Parameter list should have remainder 1 mod 6")

    ngaussians = (len(p) - 1) // 6

    val = 0
    for ii in range(ngaussians - 1):
        ps = np.concatenate((np.array(p[6*ii: 6*ii + 5]), np.array([0]), np.atleast_1d([p[ii * 6 + 5]])))
        val += gauss_fn(x, y, ps)

    # deal with last gaussian, which also gets background term
    ps = np.concatenate((np.array(p[-7:-2]), np.atleast_1d(p[-1]), np.atleast_1d(p[-2])))
    val += gauss_fn(x, y, ps)
    return val


def sum_gauss_jacobian(x, y, p):
    """
    Jacobian of the sum of n 2D gaussians
    :param x:
    :param y:
    :param p:
    :return:
    """
    if len(p) % 6 != 1:
        raise ValueError("Parameter array had wrong length")

    ngaussians = (len(p) - 1) // 6

    jac_list = []
    for ii in range(ngaussians - 1):
        ps = np.concatenate((np.array(p[6 * ii: 6 * ii + 5]), np.array([0]), np.atleast_1d([p[ii * 6 + 5]])))
        jac_current = gauss_jacobian(x, y, ps)
        jac_list += jac_current[:-2] + [jac_current[-1]]

    # deal with last gaussian, which also gets background term
    ps = np.concatenate((np.array(p[-7:-2]), np.atleast_1d(p[-1]), np.atleast_1d(p[-2])))
    jac_current = gauss_jacobian(x, y, ps)
    jac_list += jac_current[:-2] + [jac_current[-1]] + [jac_current[-2]]

    return jac_list


# other functions
def poly(xx, yy, params, max_orders=(1, 1)):
    """

    :param xx:
    :param yy:
    :param params: [cx, cy, b00, b01, ..., b0N, b10, ...bMN], where
    P(x) = \sum_{n,m} cnm (X - cx)**n * (Y - cy)**m
    :param max_orders:
    :return:
    """

    max_x, max_y = max_orders
    ny = max_y + 1
    nx = max_x + 1
    norders = nx * ny

    if len(params) != (norders + 2):
        raise ValueError('params is not equal to norders')

    cx = params[0]
    cy = params[1]

    val = 0
    for ii in range(nx):
        for jj in range(ny):
            val += params[2 + ii * ny + jj] * (xx - cx) ** ii * (yy - cy) ** jj

    return val
