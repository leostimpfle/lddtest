import math
import typing
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

from lddtest.utils import round_to_integer

# https://www.sciencedirect.com/science/article/pii/S0304407607001133
# https://eml.berkeley.edu/~jmccrary/DCdensity/
# https://github.com/ddimmery/rdd/blob/master/DCdensity.R

def dcdensity(
        running: np.typing.ArrayLike,
        cutoff: float = 0.0,
        bin_size: typing.Optional[float] = None,
        bandwidth: typing.Optional[float] = None,
        do_plot: bool = False,
) -> pd.Series:
    N = running.shape[0]
    running_std = np.std(running)
    running_min = np.min(running)
    running_max = np.max(running)

    if cutoff <= running_min or cutoff >= running_max:
        raise ValueError(
            'Cutoff must lie within range of running variable.'
        )

    if bin_size is None:
        bin_size = 2 * running_std * N**-0.5  # p. 705 (McCrary, 2008)

    left_bin = _get_midpoint(r=running_min, bin_size=bin_size, cutoff=cutoff) # midpoint of lowest bin
    right = _get_midpoint(r=running_max, bin_size=bin_size, cutoff=cutoff)  # midpoint of highest bin
    left_cut = cutoff - bin_size / 2  # midpoint of bin just left of cutoff
    right_cut = cutoff + bin_size / 2  # midpoint of bin just right of cutoff
    j = math.floor((running_max - running_min) / bin_size) + 2
    bin_numbers = _get_bin_numbers(
        running=running,
        cutoff=cutoff,
        bin_size=bin_size,
        left=left_bin,
    )
    # counts of observations in each cell
    bin_counts = np.zeros(j, dtype=float)
    values, counts = np.unique_counts(bin_numbers)
    bin_counts[values-1] = counts
    bin_counts /= N  # convert counts to fraction
    bin_counts /= bin_size  # normalize histogram to integrate to 1
    # calculate midpoint of cell
    bin_midpoints = np.floor(
        (
            left_bin
            + (np.arange(start=1, stop=j+1, step=1, dtype=int) - 1) * bin_size
            - cutoff
        ) / bin_size
    ) * bin_size + bin_size / 2 + cutoff

    if bandwidth is None:
        # calculate bandwidth
        bin_left = round_to_integer(
            (
                ((math.floor((left_cut - cutoff) / bin_size) * bin_size + bin_size / 2 + cutoff) - left_bin)
                / bin_size
            ) + 1
        )
        bin_right = round_to_integer(
            (
                ((math.floor((right_cut - cutoff) / bin_size) * bin_size + bin_size / 2 + cutoff) - left_bin)
                / bin_size
            ) + 1
        )
        if bin_right - bin_left != 1:
            raise ValueError()

        cell_midpoints_left = bin_midpoints[:bin_left]
        cell_midpoints_right = bin_midpoints[bin_right:]

        # estimate 4th order polynomial to the left
        data = pd.DataFrame(
            data=np.vstack((bin_counts, bin_midpoints)).T,
            columns=['cells', 'midpoints'],
        )
        subsets = {
            left_bin: (cell_midpoints_left, bin_midpoints < cutoff),
            right: (cell_midpoints_right, bin_midpoints >= cutoff),
        }
        bandwidth = [
            _get_bandwidth(
                midpoint=midpoint,
                midpoints=midpoints,
                data=data,
                subset=subset,
                cutoff=cutoff,
            )
            for midpoint, (midpoints, subset) in subsets.items()
        ]
        bandwidth = sum(bandwidth) / 2

    observations_left = (running > cutoff - bandwidth) & (running < cutoff)
    observations_right = (running < cutoff + bandwidth) & (running >= cutoff)
    if not observations_left.any() or not observations_right.any():
        ValueError('Insufficient data within the bandwidth.')

    if do_plot:
        raise NotImplementedError

    # add padding zeros to histogram (to assist smoothing)
    padzeros = math.ceil(bandwidth / bin_size)
    jp = j + 2 * padzeros
    if padzeros >= 1:
        bin_counts_padded = np.concatenate(
            (
                np.zeros(padzeros),
                bin_counts,
                np.zeros(padzeros),
            )
        )
        bin_midpoints_padded = np.concatenate(
            (
                np.arange(
                    start=left_bin - padzeros * bin_size,
                    stop=left_bin,
                    step=bin_size,
                ),
                bin_midpoints,
                np.arange(
                    start=right + bin_size,
                    stop=right + padzeros * bin_size + bin_size,
                    step=bin_size,
                ),
            )
        )
    else:
        bin_counts_padded = bin_counts
        bin_midpoints_padded = bin_midpoints

    # estimate density to the left
    distance = bin_midpoints_padded - cutoff
    weights = 1 - abs(distance / bandwidth)
    weights = np.where(
        weights > 0,
        weights * (bin_midpoints_padded < cutoff),
        0,
    )
    weights = weights / weights.sum() * jp
    density_left = sm.WLS(
        endog=bin_counts_padded,
        exog=sm.add_constant(distance),
        weights=weights,
    ).fit()
    density_left_estimate = density_left.predict([1, 0]).squeeze().item()

    # estimate density to the right
    weights = 1 - abs(distance / bandwidth)
    weights = np.where(
        weights > 0,
        weights * (bin_midpoints_padded >= cutoff),
        0,
    )
    weights = weights / weights.sum() * jp
    density_right = sm.WLS(
        endog=bin_counts_padded,
        exog=sm.add_constant(distance),
        weights=weights,
    ).fit()
    density_right_estimate = density_right.predict([1, 0]).squeeze().item()

    # estimate density discontinuity
    theta_hat = (
            math.log(density_right_estimate)
            - math.log(density_left_estimate)
    )
    # equation 5 (McCrary, 2008)
    theta_hat_se = math.sqrt(
        1/(N * bandwidth) * 24/5
        * (1/density_right_estimate + 1/density_left_estimate)
    )
    z_stat = theta_hat / theta_hat_se
    p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))

    results = pd.Series(
        {
            'density discontinuity (log difference)': theta_hat,
            'density discontinuity (standard error)': theta_hat_se,
            'z-statistic': z_stat,
            'p-value': p_value,
            'bandwidth': bandwidth,
            'bin size': bin_size,
            'cutoff': cutoff,
        },
        name='results'
    )
    return results


def _get_bandwidth(
        midpoint: float,
        midpoints: np.typing.ArrayLike,
        data: pd.DataFrame,
        subset: np.typing.ArrayLike,
        cutoff: float,
        endogenous: str = 'cells',
        exogenous: str = 'midpoints',
        degree: int = 4,
        kappa: float = 3.348,
) -> float:
    # p. 705 (McCrary, 2008)
    degrees = range(1, degree + 1)
    regressors = [
        f'I({exogenous} ** {order})' for order in degrees
    ]
    formula = f'{endogenous} ~ {"+".join(regressors)}'
    model = smf.ols(
        formula=formula,
        data=data,
        subset=subset,
    )
    fit = model.fit()
    second_derivative = np.array(
        [
            math.factorial(order)
            / math.factorial(order - 2)
            * fit.params[f'I({exogenous} ** {order})']
            for order in degrees
            if order > 1
        ]
    )
    xs = np.vstack(
        [
            midpoints**(order-2)
            for order in degrees if order > 1
        ]
    )
    bandwidth = kappa * (
        fit.mse_model * abs(cutoff - midpoint)
        / np.square(second_derivative[:, None] * xs).sum()
    ) ** (1/5)
    return bandwidth


def _get_midpoint(
        r: np.typing.ArrayLike,
        bin_size: float,
        cutoff: float,
):
    return np.floor(
        (r - cutoff) / bin_size
    ) * bin_size + bin_size / 2 + cutoff

def _get_bin_numbers(
        running: np.typing.ArrayLike,
        cutoff: float,
        bin_size: float,
        left: float,
) -> np.typing.ArrayLike:
    # equation 2 (McCrary, 2008)
    bin_numbers = round_to_integer(
        (_get_midpoint(r=running, cutoff=cutoff, bin_size=bin_size) - left)
        / bin_size
        + 1
    )
    return bin_numbers
