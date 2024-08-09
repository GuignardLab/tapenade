import numpy as np
import numba
from numba.typed import List
from itertools import product

"""
The code for fast interpolation takes inspiration from the library fast_interp
(https://github.com/dbstein/fast_interp) from David Stein.
"""


################################################################################
# variables to control when we switch between serial / parallel versions

serial_cutoffs = [0, 2000, 400, 100]


def set_serial_cutoffs(dimension, cutoff):
    serial_cutoffs[dimension] = cutoff


def _compute_bounds1(a, b, h, p, c, e, k):
    if p:
        return -1e100, 1e100
    elif not c:
        d = h * (k // 2)
        return a + d, b - d
    else:
        d = e * h
        u = b + d
        u -= (
            u * 1e-15
        )  # the routines can fail when we exactly hit the right endpoint, this protects against that
        return a - d, u


def _compute_bounds(a, b, h, p, c, e, k):
    m = len(a)
    bounds = [
        _compute_bounds1(a[i], b[i], h[i], p[i], c[i], e[i], k)
        for i in range(m)
    ]
    return [list(x) for x in zip(*bounds)]


def _fill3(f, fb, ox, oy, oz):
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]
    if nx * ny * nz < 100000:
        fb[ox : ox + nx, oy : oy + ny, oz : oz + nz] = f
    else:
        __fill3(f, fb, ox, oy, oz)


@numba.njit(parallel=True)
def __fill3(f, fb, ox, oy, oz):
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]
    for i in numba.prange(nx):
        for j in range(ny):
            for k in range(nz):
                fb[i + ox, j + oy, k + oz] = f[i, j, k]


################################################################################
# utilities to enable serial / parallel compilation of same function


def sjit(func):
    return numba.njit(func, parallel=False)


def pjit(func):
    return numba.njit(func, parallel=True)


################################################################################
# utility to allow construction of TypedList with Float/Int types (always promoted to float)


def FloatList(l):
    return List([float(lh) for lh in l])


def IntList(l):
    return List([int(lh) for lh in l])


def BoolList(l):
    return List([bool(lh) for lh in l])


################################################################################
# 1D Extrapolation Routines


def _extrapolate1d_x(f, o):
    for ix in range(o):
        il = o - ix - 1
        ih = f.shape[0] - (o - ix)
        f[il] = 2 * f[il + 1] - 1 * f[il + 2]
        f[ih] = 2 * f[ih - 1] - 1 * f[ih - 2]


def _extrapolate1d_y(f, o):
    for ix in range(o):
        il = o - ix - 1
        ih = f.shape[1] - (o - ix)
        f[:, il] = 2 * f[:, il + 1] - 1 * f[:, il + 2]
        f[:, ih] = 2 * f[:, ih - 1] - 1 * f[:, ih - 2]


def _extrapolate1d_z(f, o):
    for ix in range(o):
        il = o - ix - 1
        ih = f.shape[2] - (o - ix)
        f[:, :, il] = 2 * f[:, :, il + 1] - 1 * f[:, :, il + 2]
        f[:, :, ih] = 2 * f[:, :, ih - 1] - 1 * f[:, :, ih - 2]


################################################################################


# interpolation routines
def _interp3d_k1(f, xout, yout, zout, fout, a, h, n, p, o, lb, ub):
    m = fout.shape[0]
    for mi in numba.prange(m):
        xr = min(max(xout[mi], lb[0]), ub[0])
        yr = min(max(yout[mi], lb[1]), ub[1])
        zr = min(max(zout[mi], lb[2]), ub[2])
        xx = xr - a[0]
        yy = yr - a[1]
        zz = zr - a[2]
        ix = int(xx // h[0])
        iy = int(yy // h[1])
        iz = int(zz // h[2])
        ratx = xx / h[0] - (ix + 0.5)
        raty = yy / h[1] - (iy + 0.5)
        ratz = zz / h[2] - (iz + 0.5)
        asx = np.empty(2)
        asy = np.empty(2)
        asz = np.empty(2)
        asx[0] = 0.5 - ratx
        asx[1] = 0.5 + ratx
        asy[0] = 0.5 - raty
        asy[1] = 0.5 + raty
        asz[0] = 0.5 - ratz
        asz[1] = 0.5 + ratz
        ix += o[0]
        iy += o[1]
        iz += o[2]
        fout[mi] = 0.0
        for i in range(2):
            ixi = (ix + i) % n[0] if p[0] else ix + i
            for j in range(2):
                iyj = (iy + j) % n[1] if p[1] else iy + j
                for k in range(2):
                    izk = (iz + k) % n[2] if p[2] else iz + k
                    if not (np.isnan(f[ixi, iyj, izk])):
                        fout[mi] += f[ixi, iyj, izk] * asx[i] * asy[j] * asz[k]


_s_interp3d_k1 = sjit(_interp3d_k1)
_p_interp3d_k1 = pjit(_interp3d_k1)


def _extrapolate3d(f, k, p, c, e):
    padx = (not p[0]) and c[0]
    pady = (not p[1]) and c[1]
    padz = (not p[2]) and c[2]
    if padx or pady or padz:
        ox = (k // 2) + e[0] if padx else 0
        oy = (k // 2) + e[1] if pady else 0
        oz = (k // 2) + e[2] if padz else 0
        fb = np.zeros(
            [f.shape[0] + 2 * ox, f.shape[1] + 2 * oy, f.shape[2] + 2 * oz],
            dtype=f.dtype,
        )
        _fill3(f, fb, ox, oy, oz)
        if padx:
            _extrapolate1d_x(fb, ox)
        if pady:
            _extrapolate1d_y(fb, oy)
        if padz:
            _extrapolate1d_z(fb, oz)
        return fb, [ox, oy, oz]
    else:
        return f, [0, 0, 0]
    return fb


class interp3d(object):
    def __init__(self, a, b, h, f, p=[False] * 3, c=[True] * 3, e=[0] * 3):
        """
        See the documentation for interp1d
        this function is the same, except that a, b, h, p, c, and e
        should be lists or tuples of length 3 giving the values for each
        dimension
        the function behaves as in the 1d case, except that of course padding
        is required if padding is requested in any dimension
        """
        # interpolation order
        k = 1

        self.a = FloatList(a)
        self.b = FloatList(b)
        self.h = FloatList(h)
        self.f = f
        self.k = k
        self.p = BoolList(p)
        self.c = BoolList(c)
        self.e = IntList(e)
        self.n = IntList(f.shape)
        self.dtype = f.dtype
        self._f, _o = _extrapolate3d(f, k, p, c, e)
        self._o = IntList(_o)
        lb, ub = _compute_bounds(a, b, h, p, c, e, k)
        self.lb = FloatList(lb)
        self.ub = FloatList(ub)

    def __call__(self, xout, yout, zout, fout=None):
        """
        Interpolate to xout
        For 1-D interpolation, xout must be a float
            or a ndarray of floats
        """
        if isinstance(xout, np.ndarray):
            if xout.size > serial_cutoffs[3]:
                func = _p_interp3d_k1
            else:
                func = _s_interp3d_k1
            m = int(np.prod(xout.shape))
            copy_made = False
            if fout is None:
                _out = np.empty(m, dtype=self.dtype)
            else:
                _out = fout.ravel()
                if _out.base is None:
                    copy_made = True
            _xout = xout.ravel()
            _yout = yout.ravel()
            _zout = zout.ravel()

            func(
                self._f,
                _xout,
                _yout,
                _zout,
                _out,
                self.a,
                self.h,
                self.n,
                self.p,
                self._o,
                self.lb,
                self.ub,
            )
            if copy_made:
                fout[:] = _out
            return _out.reshape(xout.shape)
        else:
            func = _s_interp3d_k1
            _xout = np.array(
                [xout],
            )
            _yout = np.array(
                [yout],
            )
            _zout = np.array(
                [zout],
            )
            _out = np.empty(1)
            func(
                self._f,
                _xout,
                _yout,
                _zout,
                _out,
                self.a,
                self.h,
                self.n,
                self.p,
                self._o,
                self.lb,
                self.ub,
            )
            return _out[0]


################################################################################


def _local_equalization(
    image: np.ndarray, box_size: int, 
    perc_low: float, perc_high: float,
    mask: np.ndarray = None
) -> np.ndarray:
    """
    Performs local histogram stretching by applying the following steps:
        1. compute the percentile values in a neighborhood around voxels chosen on a regular grid.
        2. interpolate these values on each voxel of the image.
        3. perform a linear equalization between the interpolated percentile values and the values 0 and 1.
        4. clip the image voxel values between 0 and 1.

    Parameters
    ----------
    image : np.ndarray
        The image to equalize.
    box_size : int
        size of the neighborhood box around each voxel to compute the percentile values.
    perc_low : float
        percentile value to use as the low value of the equalization (will be mapped to 0).
    perc_high : float
        percentile value to use as the high value of the equalization (will be mapped to 1).
    mask : np.ndarray
        An optional mask boolean array from which background values will be excluded from the computation.

    Returns
    -------
    image_norm : np.ndarray
        The equalized image.
    """

    # Compute necessary variables
    array_shape = image.shape
    box_length = 2 * box_size + 1
    grid_shape = [int(np.ceil(s / box_length)) + 1 for s in array_shape]
    grid_positions_and_steps = [
        np.linspace(0, s - 1, n_boxes, retstep=True)
        for s, n_boxes in zip(array_shape, grid_shape)
    ]

    grid_positions = [positions for positions, _ in grid_positions_and_steps]
    grid_steps = [step for _, step in grid_positions_and_steps]

    percs_low = np.zeros(grid_shape)
    percs_high = np.zeros(grid_shape)

    percs_positions = [np.arange(s) for s in grid_shape]

    # Compute percentile values for each neighborhood
    for indices_grid, indices_percs in zip(
        product(*grid_positions), product(*percs_positions)
    ):
        indices_grid = np.array(indices_grid).round().astype(int)

        slices = []
        for index, s in zip(indices_grid, array_shape):
            start = min(max(index - box_size, 0), s - box_length)
            stop = min(max(index + box_size + 1, box_length), s)
            slices.append(slice(start, stop))

        box = image[tuple(slices)].copy()

        if mask is not None:
            box = box[mask[tuple(slices)]]
            if box.size == 0:
                continue

        val_low, val_high = np.percentile(box, [perc_low, perc_high])

        percs_low[tuple(indices_percs)] = val_low
        percs_high[tuple(indices_percs)] = val_high

    full_positions = np.array(
        np.meshgrid(*[np.arange(s) for s in array_shape], indexing="ij")
    ).reshape(image.ndim, -1)

    # Interpolate percentile values for each voxel
    interp_low = interp3d(
        a=[0] * image.ndim,
        b=[s - 1 for s in image.shape],
        h=grid_steps,
        f=percs_low,
        c=[False] * image.ndim,
    )

    interp_high = interp3d(
        a=[0] * image.ndim,
        b=[s - 1 for s in image.shape],
        h=grid_steps,
        f=percs_high,
        c=[False] * image.ndim,
    )

    full_percs_low = interp_low(*full_positions).reshape(array_shape)
    full_percs_high = interp_high(*full_positions).reshape(array_shape)

    denom = full_percs_high - full_percs_low
    denom[denom == 0] = 1
    denom += 1e-10

    # Perform linear equalization and clip image voxel values
    image_norm = np.clip((image - full_percs_low) / denom, 0, 1)

    return image_norm
