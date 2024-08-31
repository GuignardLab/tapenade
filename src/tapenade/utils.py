import numpy as np
from scipy.optimize import least_squares
from scipy.stats import linregress


def filter_percentiles(
    X, percentilesX: tuple = (1, 99), Y=None, percentilesY: tuple = None
):

    if Y is None:

        down, up = percentilesX

        percentile_down = np.percentile(X, down)
        percentile_up = np.percentile(X, up)

        mask = np.logical_and(percentile_down < X, percentile_up > X)

        return X[mask]

    else:

        downX, upX = percentilesX

        if percentilesY is None:
            downY, upY = percentilesX
        else:
            downY, upY = percentilesY

        percentile_downX = np.percentile(X, downX)
        percentile_downY = np.percentile(Y, downY)

        percentile_upX = np.percentile(X, upX)
        percentile_upY = np.percentile(Y, upY)

        maskX = np.logical_and(percentile_downX <= X, percentile_upX >= X)
        maskY = np.logical_and(percentile_downY <= Y, percentile_upY >= Y)

        mask = np.logical_and(maskX, maskY)

        return X[mask], Y[mask]


def linear_fit(
    x,
    y,
    robust: bool = False,
    return_r2: bool = False,
    robust_params_init: tuple = None,
    robust_f_scale: float = None,
):

    if not robust:
        res = linregress(x, y)

        if return_r2:
            return res.intercept, res.slope, res.rvalue**2
        else:
            return res.intercept, res.slope

    else:

        def f(params, x, y):
            return params[0] + params[1] * x - y

        if robust_params_init is None:
            robust_params_init = np.ones(2)

        res_robust = least_squares(
            f,
            robust_params_init,
            args=(x, y),
            loss="soft_l1",
            f_scale=robust_f_scale,
        )

        if return_r2:
            raise NotImplementedError
        else:
            return res_robust.x[0], res_robust.x[1]
