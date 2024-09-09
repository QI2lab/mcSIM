"""
Generate DMD patterns for FS-ODT patterns. A typical workflow is to choose (1) number of ODT angles (2) degree of
multiplexing and (3) degree of position multiplexing and positions. Then, use get_odt_spot_locations() to choose
positions in the pupil. Follow this with get_multiplexed_spot_positions() to combine these into angle multiplexed
patterns. Next, introduce position multiplexing with add_position_multiplexing(). Finally generate the desired patterns
with get_odt_patterns()
"""
from time import perf_counter
from collections.abc import Sequence
import numpy as np


def get_odt_spot_locations(nmax: int,
                           fmax: float = 0.95,
                           fx_max: float = 1.,
                           fy_max: float = 1.) -> np.ndarray:
    """
    Generate set of spot locations equally spaced in back-pupil plane. Scale is set by fmax

    :param nmax: maximum number of spots
    :param fmax: maximum radius
    :param fx_max: maximum distance in x-direction
    :param fy_max: maximum distance in y-direction
    :return centers: ncenters x 2 in order cx x cy
    """

    # if no xy restrictions, circle
    area = np.pi * fmax**2

    # remove area of section of circle beyond fx_max
    if fx_max < fmax:
        theta_x = np.arccos(fx_max / fmax)
        # area of portion of circle but subtract two triangles
        area_diff_x = np.pi * fmax**2 * (theta_x / np.pi) - np.sqrt(fmax**2 - fx_max**2) * fx_max
        area -= 2 * area_diff_x

    if fy_max < fmax:
        theta_y = np.arccos(fy_max / fmax)
        area_diff_y = np.pi * fmax ** 2 * (theta_y / np.pi) - np.sqrt(fmax ** 2 - fy_max ** 2) * fy_max
        area -= 2 * area_diff_y

    # overlap between areas we subtracted
    if fx_max < fmax and fy_max < fmax:
        # looking at one quadrant
        # 0.5 * Ax + 0.5 * Ay + fx_max * fy_max = 0.25 * np.pi * fmax^2 + overlap
        overlap = 0.5 * area_diff_x + 0.5 * area_diff_y + fx_max * fy_max - 0.25 * np.pi * fmax**2
        area += 4 * overlap

    df = np.sqrt(area / nmax)
    stop = False

    while not stop:
        cx = np.concatenate((np.flip(-np.arange(df, fx_max + df, df)),
                             np.array([0]),
                             np.arange(df, fx_max + df, df)
                             ))
        cy = np.concatenate((np.flip(-np.arange(df, fy_max + df, df)),
                             np.array([0]),
                             np.arange(df, fy_max + df, df)
                             ))

        cxcx, cycy = np.meshgrid(cx, cy)
        cc = np.sqrt(cxcx**2 + cycy**2)

        to_use = np.logical_and.reduce((cc <= fmax,
                                        cxcx <= fx_max,
                                        cxcx >= -fx_max,
                                        cycy <= fy_max,
                                        cycy >= -fy_max))

        # todo ... in principle should find first region with max freqs
        # todo then keep expanding as long as number does not decrease to try and reach edge
        if np.sum(to_use) > nmax:
            df *= 1.01
        else:
            stop = True
            centers = np.stack((cxcx[to_use], cycy[to_use]), axis=1)

    return centers


def get_multiplexed_spot_positions(centers: np.ndarray,
                                   n_multiplex: int,
                                   max_dist: float = 0.25,
                                   n_swap_cycles: int = 5,
                                   n_swaps_per_pair: int = 300,
                                   verbose: bool = False
                                   ) -> list[np.ndarray]:
    """
    Generate multiplexed spot positions from a list of initial spot positions
    Try and optimize this set so that no spots are near each other

    :param centers: Spot pattern center positions in the pupil. npatterns x 2 array. Typically centers are generated
      from get_odt_spot_locations()
    :param n_multiplex: number of centers to multiplex
    :param max_dist: spot distances above this threshold experience the same penalty
    :param n_swap_cycles: number of times to loop through pairs of patterns
    :param n_swaps_per_pair: for each pair of patterns, randomly swap this many spot positions and search for lower cost
    :param verbose: print progress of swapping algorithm
    :return center_sets:
    """

    # generate first guess at steps
    ncenters = len(centers)
    npatterns = int(np.ceil(len(centers) / n_multiplex))
    centers_used_now = np.zeros(len(centers), dtype=bool)

    center_sets = []
    for ii in range(npatterns):

        # not multiplexing
        if n_multiplex == 1:
            centers_now = centers[ii][None, :]
        else:
            centers_now = np.zeros([n_multiplex, 2]) * np.nan

            for jj in range(n_multiplex):
                if jj == 0:
                    # get next unused index
                    ind = np.min(np.arange(ncenters)[np.logical_not(centers_used_now)])
                else:
                    # compute furthest remaining ...
                    dists = np.linalg.norm(centers_now[:, None, :] - centers[None, :, :], axis=-1)

                    # only scale distance penalty up to certain maximum distance
                    dists[dists >= max_dist] = max_dist

                    # sum distance from current points
                    dist_sum = np.nansum(dists, axis=0)

                    # penalty for points lying on the same line
                    if n_multiplex != 3 or jj < 2:
                        line_penalty = 0
                    else:
                        slope, pt_line = _nearest_line_pts(centers_now[:jj])

                        _, d = _nearest_pt_line(centers, slope, pt_line)
                        line_penalty = np.mean(d)

                        # todo: fix bug when slope -> inf
                        d[np.isnan(d)] = 0

                    # exclude any points already found (unless there aren't any left)
                    nan_mask = np.ones(centers_used_now.shape)
                    if np.any(np.logical_not(centers_used_now)):
                        nan_mask[centers_used_now] = np.nan

                    ind = np.nanargmax((dist_sum + line_penalty) * nan_mask)

                centers_used_now[ind] = True
                centers_now[jj] = centers[ind]

        center_sets.append(centers_now)

        # re-cycle through first indices if necessary to ensure all patterns at a given level
        # of multiplexing have the same number of frequencies

    def loss(c, include_line_penalty=True):
        d = np.linalg.norm(c - c[:, None, :], axis=-1)
        d[d > max_dist] = max_dist

        # distance from line
        if include_line_penalty:
            slope, pt_line = _nearest_line_pts(c)
            _, d_line = _nearest_pt_line(c, slope, pt_line)
            line_penalty = np.sum(d_line)
        else:
            line_penalty = 0

        return np.sum(d) + line_penalty

    # swap frequencies between patterns to achieve better cost
    tstart_swap = perf_counter()
    if n_multiplex > 1:
        for aa in range(n_swap_cycles):
            if verbose:
                print(f"swap {aa + 1:d}/{n_swap_cycles:d}, "
                      f"elapsed time = {perf_counter() - tstart_swap:.2f}", end="\r")

            for ii in range(npatterns):
                for jj in range(npatterns):
                    if ii == jj:
                        continue

                    for _ in range(n_swaps_per_pair):
                        p1 = center_sets[ii]
                        p2 = center_sets[jj]

                        cstart = loss(p1) + loss(p2)

                        # choose random integers and swap patterns
                        rand1 = np.random.randint(0, len(p1))
                        rand2 = np.random.randint(0, len(p2))

                        p1_prop = np.array(p1, copy=True)
                        p1_prop[rand1] = p2[rand2]

                        p2_prop = np.array(p2, copy=True)
                        p2_prop[rand2] = p1[rand1]

                        cprop = loss(p1_prop) + loss(p2_prop)

                        # if swap improved things, keep it
                        if cprop > cstart:
                            center_sets[ii] = p1_prop
                            center_sets[jj] = p2_prop

        print("")

    return center_sets


def add_position_multiplexing(centers: list[np.ndarray],
                              frqs: np.ndarray,
                              ) -> (list[np.ndarray], list[np.ndarray]):
    """
    Given a set of spot positions (which may or may not include angle multiplexing), generate a set of
    frequency multiplexed patterns. To do this, start from our original set of patterns, and
    create new patterns by combining len(frqs) adjacent patterns using different frequencies

    :param centers: list of spot center positions for each pattern. Each element of the list should be a
      nangles x 2 array. nangles can be different for each image. Typically, centers is generated by
      get_multiplexed_spot_locations()
    :param frqs: nfrqs x 2 array
    :return centers_out, frqs_out: lists of length npatterns. Each element of each list is of size
      (nangles * len(frqs)) x 2.
    """

    if len(centers) < len(frqs):
        raise ValueError(f"cannot position multiplex because total "
                         f"number of spot positions = {len(centers):d} < "
                         f"number of spot frequencies = {len(frqs):d}")

    centers_multi = []
    frqs_multi = []
    for ii in range(len(centers)):
        # grab len(frqs) adjacent images ... we will use different frequencies for each
        imgs_now = centers[ii: ii + len(frqs)]
        if len(imgs_now) < len(frqs):
            imgs_now += centers[:len(frqs) - len(imgs_now)]
        
        # assign unique frequency to each image, and get list of frequency arrays the same size as centers
        frqs_now = []
        for jj in range(len(frqs)):
            frqs_now.append(np.tile(frqs[jj], [len(imgs_now[jj]), 1]))

        # append data
        centers_multi.append(np.concatenate(imgs_now, axis=0))
        frqs_multi.append(np.concatenate(frqs_now, axis=0))

    return centers_multi, frqs_multi


def _nearest_pt_line(pt,
                     slope,
                     pt_line):
    """
    Get the shortest distance between a point and a line.

    :param pt: (xo, yo), point of itnerest
    :param slope: slope of line
    :param pt_line: (xl, yl), point the line passes through
    :return pt: (x_near, y_near), nearest point on the line.
    :return d: shortest distance from point to line.
    """
    # xo, yo = pt
    xos = pt[..., 0]
    yos = pt[..., 1]
    xl, yl = pt_line
    b = yl - slope * xl

    x_int = (xos + slope * (yos - b)) / (slope**2 + 1)
    y_int = slope * x_int + b
    d = np.sqrt((xos - x_int)**2 + (yos - y_int)**2)

    return (x_int, y_int), d


def _nearest_line_pts(pts):
    """
    Get the closest line fitting a set of points

    :param pts:
    :return slope, pt_line:
    """
    xs = pts[:, 0]
    ys = pts[:, 1]

    a = np.concatenate((np.ones((len(pts), 1)), xs[:, None]), axis=1)
    lsq_params, _, _, _ = np.linalg.lstsq(a,
                                          ys,
                                          rcond=None)

    offset, slope = lsq_params
    pt_line = np.array([0, offset])

    return slope, pt_line


def get_odt_patterns(pupil_positions: Sequence[np.ndarray],
                     dmd_size: Sequence[int, int],
                     spot_radius: float,
                     pupil_radius_mirrors: float,
                     frqs: Sequence[np.ndarray],
                     phase: float = 0.,
                     use_off_mirrors: bool = True) -> (np.ndarray, list[dict]):
    """
    Generate DMD patterns from a list of center positions

    :param pupil_positions: list of arrays, where each array is of size N x 2 array in order (cx, cy)
    :param dmd_size: (ny, nx)
    :param spot_radius: spot radius in mirrors
    :param pupil_radius_mirrors: typically this is calculated using
      focal_len_detection * na_detection / mag_dmd2bfp / dm
    :param frqs: list of arrays, where each array is the same size as the corresponding entry in pupil_positions
      carrier frequencies (fx, fy) in 1/mirrors. This should be a list of arrays of the same
      size as pupil_positions
    :param phase: phase of carrier frequency pattern
    :param use_off_mirrors:
    :return odt_patterns, odt_pattern_data:
    """

    if len(frqs) != len(pupil_positions):
        frqs_old = np.array(frqs, copy=True)
        frqs = [np.zeros(pupil_positions[ii].shape) + frqs_old for ii in range(len(pupil_positions))]

    ny, nx = dmd_size
    npatterns = len(pupil_positions)
    xx, yy = np.meshgrid(range(nx), range(ny))

    if use_off_mirrors:
        odt_patterns = np.ones((npatterns, ny, nx), dtype=bool)
    else:
        odt_patterns = np.zeros((npatterns, ny, nx), dtype=bool)

    # loop over patterns
    odt_pattern_data = []
    for ii in range(len(pupil_positions)):
        frqs_mirrors = frqs[ii]
        spot_pos_mirrors = pupil_positions[ii] * pupil_radius_mirrors + np.array([[nx // 2, ny // 2]])

        if len(spot_pos_mirrors) != len(frqs_mirrors):
            raise ValueError("pupil positions must match size of frqs")

        # loop over centers and create spots
        for kk in range(len(spot_pos_mirrors)):
            to_use = np.sqrt((xx - spot_pos_mirrors[kk, 0]) ** 2 + (yy - spot_pos_mirrors[kk, 1]) ** 2) <= spot_radius

            pnow = np.round(np.cos(2 * np.pi * (xx[to_use] * frqs_mirrors[kk, 0] +
                                                yy[to_use] * frqs_mirrors[kk, 1]) + phase), 12)

            pnow[pnow <= 0] = 0
            pnow[pnow > 0] = 1

            if use_off_mirrors:
                pnow = 1 - pnow

            odt_patterns[ii, to_use] = pnow

        odt_pattern_data.append({"type": "odt",
                                 "spot_frqs_mirrors": frqs_mirrors.tolist(),
                                 "nposition_multiplex": np.unique(frqs_mirrors, axis=0).shape[0],
                                 "spot_positions_mirrors": spot_pos_mirrors.tolist(),
                                 "nangles_multiplex_nominal": len(spot_pos_mirrors),
                                 "phase": phase,
                                 "radius": spot_radius,  # carrier frequency information
                                 "pupil_frequency_fraction": np.linalg.norm(pupil_positions[ii], axis=1).tolist(),
                                 "pupil_angle": np.arctan2(spot_pos_mirrors[:, 1], spot_pos_mirrors[:, 0]).tolist()})

    return odt_patterns, odt_pattern_data


def get_subset(centers: np.ndarray,
               n_subset: int,
               fraction: float,
               df: float = 0.1,
               niters: int = 100):
    """
    Get subset of n_subset patterns which are near radius fraction and roughly equally spaced

    :param centers:
    :param n_subset:
    :param fraction:
    :param df: consider frequencies within this distance of fraction
    :param niters: number of iterations to swap centers
    :return inds: indices of centers in full center array
    """

    fracs = np.linalg.norm(centers, axis=1)

    allowed = np.abs(fracs - fraction) <= df

    inds = np.arange(len(centers), dtype=int)[allowed]

    def loss(c):
        dists = np.linalg.norm(c - c[:, None, :], axis=-1)

        return np.sum(np.mean(dists))

    used_now = np.zeros(len(inds), dtype=bool)

    if n_subset >= len(inds):
        inds_now = inds
    else:
        # initialize
        inds_now = inds[:n_subset]
        used_now[:n_subset] = True
        for ii in range(niters):
            for jj in range(n_subset):
                rand = np.random.randint(0, len(inds) - n_subset)

                inds_proposed = np.array(inds_now, copy=True)
                inds_proposed[jj] = inds[np.logical_not(used_now)][rand]

                l1 = loss(centers[inds_now])
                l2 = loss(centers[inds_proposed])

                if l2 > l1:
                    used_now[inds == inds_proposed[jj]] = True
                    used_now[inds == inds_now[jj]] = False

                    inds_now = inds_proposed

    return inds_now
