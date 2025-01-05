import math
import typing
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
try:
    import plotly.graph_objects as go
    has_plotly = True
except ModuleNotFoundError:
    has_plotly = False

from lddtest.enums import DcdensityResults
from lddtest.kernel import Triangular
from lddtest.utils import round_to_integer

# McCrary, Justin 2008: Manipulation of the running variable in the regression discontinuity design: A density test
# DOI: 10.1016/j.jeconom.2007.05.005
# Stata code: https://eml.berkeley.edu/~jmccrary/DCdensity/
# R implementation: https://github.com/ddimmery/rdd/blob/master/DCdensity.R

def dcdensity(
        running: np.typing.ArrayLike,
        cutoff: float = 0.0,
        bin_size: typing.Optional[float] = None,
        bandwidth: typing.Optional[float] = None,
        do_plot: bool = False,
        alpha: float = 0.05,
) -> typing.Tuple[pd.Series, pd.DataFrame, typing.Union[None, go.Figure]]:
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
    right_bin = _get_midpoint(r=running_max, bin_size=bin_size, cutoff=cutoff)  # midpoint of highest bin
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
        # calculate bandwidth following Section 3.2 in (McCrary, 2008)
        # number of bin just left of cutoff
        bin_number_left = round_to_integer(
            (
                (_get_midpoint(r=left_cut, cutoff=cutoff, bin_size=bin_size) - left_bin)
                / bin_size
            ) + 1
        )
        # number of bin just right of cuttof
        bin_number_right = round_to_integer(
            (
                (_get_midpoint(r=right_cut, cutoff=cutoff, bin_size=bin_size) - left_bin)
                / bin_size
            ) + 1
        )
        if bin_number_right - bin_number_left != 1:
            raise ValueError('Bins do not align!')

        cell_midpoints_left = bin_midpoints[:bin_number_left]
        cell_midpoints_right = bin_midpoints[bin_number_right:]

        # estimate 4th order polynomial to the left
        data = pd.DataFrame(
            data=np.vstack((bin_counts, bin_midpoints)).T,
            columns=['counts', 'midpoints'],
        )
        subsets = {
            left_bin: (cell_midpoints_left, bin_midpoints < cutoff),
            right_bin: (cell_midpoints_right, bin_midpoints >= cutoff),
        }
        bandwidth = [
            _get_bandwidth(
                midpoint=midpoint,
                midpoints=midpoints,
                data=data,
                subset=subset,
                cutoff=cutoff,
                endogenous='counts',
                exogenous='midpoints',
            )
            for midpoint, (midpoints, subset) in subsets.items()
        ]
        bandwidth = sum(bandwidth) / 2

    observations_left = (running > cutoff - bandwidth) & (running < cutoff)
    observations_right = (running < cutoff + bandwidth) & (running >= cutoff)
    if not observations_left.any() or not observations_right.any():
        ValueError('Insufficient data within the bandwidth.')

    if do_plot:
        plot_data, fig = _density_plot(
            bin_midpoints=bin_midpoints,
            bin_counts=bin_counts,
            running_std=running_std,
            cutoff=cutoff,
            bandwidth=bandwidth,
            alpha=alpha,
        )
    else:
        plot_data, fig = None, None

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
                np.linspace(
                    start=left_bin - padzeros * bin_size,
                    stop=left_bin,
                    num=padzeros,
                    endpoint=True,  # include stop
                ),
                bin_midpoints,
                np.linspace(
                    start=right_bin + bin_size,
                    stop=right_bin + padzeros * bin_size,
                    num=padzeros,
                    endpoint=True,  # include stop
                )
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
    p_value = 2 * scipy.stats.norm.sf(np.abs(z_stat))

    results = pd.Series(
        {
            DcdensityResults.estimate: theta_hat,
            DcdensityResults.standard_error: theta_hat_se,
            DcdensityResults.z_stat: z_stat,
            DcdensityResults.p_value: p_value,
            DcdensityResults.bandwidth: bandwidth,
            DcdensityResults.bin_size: bin_size,
            DcdensityResults.cutoff: cutoff,
        },
        name='results'
    )
    return results, plot_data, fig


def _get_bandwidth(
        midpoint: float,
        midpoints: np.typing.ArrayLike,
        data: pd.DataFrame,
        subset: np.typing.ArrayLike,
        cutoff: float,
        endogenous: str = 'counts',
        exogenous: str = 'midpoints',
        degree: int = 4,
        kappa: float = 3.348,
) -> float:
    # p. 705 (McCrary, 2008)
    degrees = range(1, degree + 1)
    regressors = [
        f'I({exogenous} ** {order})' for order in degrees
    ]
    formula = f'{endogenous} ~ {" + ".join(regressors)}'
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
        fit.mse_resid * abs(cutoff - midpoint)
        / np.square(np.sum(second_derivative[:, None] * xs, axis=0)).sum()
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


def _density_plot(
        bin_midpoints: np.typing.ArrayLike,
        bin_counts: np.typing.ArrayLike,
        running_std: float,
        cutoff: float,
        bandwidth: float,
        alpha: float = 0.05,
) -> typing.Tuple[pd.DataFrame, typing.Union[None, go.Figure]]:
    # estimate density to either side of the cutoff using local linear regressions with triangular kernel
    selectors = (
        bin_midpoints < cutoff,
        bin_midpoints >= cutoff,
    )
    predictions = {}
    for selector in selectors:
        midpoints = bin_midpoints[selector]
        counts = bin_counts[selector]
        for i in range(midpoints.shape[0]):
            distance = midpoints - midpoints[i]
            weights = Triangular(
                x=distance,
                center=0,
                bandwidth=bandwidth,
            ).weights
            # local linear regression
            prediction = sm.WLS(
                endog=counts,
                exog=sm.add_constant(distance),
                weights=weights,
            ).fit(
                # fit model
            ).get_prediction(
                # get prediction at cutoff
                exog=[1, 0]
            ).summary_frame(
                alpha=alpha
            )
            predictions.update(
                {midpoints[i]: prediction.squeeze().rename('value')}
            )

    plot_data = pd.concat(
        predictions,
        names=['midpoint', 'variable'],
    ).unstack('variable')
    # add bin counts
    plot_data['empirical_density'] = bin_counts
    plot_data.reset_index(inplace=True)
    pmin = cutoff - 2 * running_std
    pmax = cutoff + 2 * running_std
    r = (bin_midpoints >= pmin) & (bin_midpoints <= pmax)
    if has_plotly:
        fig = _create_plotly_fig(
            plot_data=plot_data,
            selectors=selectors,
            range_x=(pmin, pmax),
            range_y=(bin_counts[r].min()*0.95, bin_counts[r].max()*1.05),
        )
    else:
        # can't produce plotly figure
        fig = None
    return plot_data, fig


def _create_plotly_fig(
        plot_data: pd.DataFrame,
        selectors: typing.Tuple[np.typing.ArrayLike, np.typing.ArrayLike],
        range_x: typing.Tuple[float, float],
        range_y: typing.Tuple[float, float],
) -> go.Figure:
    figs = [
        # empirical density
        go.Scatter(
            x=plot_data['midpoint'],
            y=plot_data['empirical_density'],
            mode='markers',
            marker=dict(color='black'),
            showlegend=False,
        ),
    ]
    for selector in selectors:
        figs.extend(
            [
                # fitted density left
                go.Scatter(
                    x=plot_data.loc[selector, 'midpoint'],
                    y=plot_data.loc[selector, 'mean'],
                    mode='lines',
                    line=dict(color='black'),
                    showlegend=False
                ),
                go.Scatter(
                    x=plot_data.loc[selector, 'midpoint'],
                    y=plot_data.loc[selector, 'mean_ci_lower'],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    showlegend=False,
                ),
                go.Scatter(
                    x=plot_data.loc[selector, 'midpoint'],
                    y=plot_data.loc[selector, 'mean_ci_upper'],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    showlegend=False
                ),
            ]
        )
    fig = go.Figure(figs)
    fig.update_xaxes(range=range_x)
    fig.update_yaxes(
        tickformat='.0%',
        range=range_y
    )
    return fig