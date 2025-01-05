import typing
import warnings

import numpy as np
from scipy.stats import norm, ncx2
from scipy.optimize import brentq

# Hartman, Erin 2021: Equivalence Testing for Regression Discontinuity Designs
# DOI: 10.1017/pan.2020.43
# Code: https://github.com/ekhartman/rdd_equivalence

def tost(
        estimate: float,
        standard_error: float,
        epsilon: float,
        alpha: float = 0.05,
) -> typing.Tuple[float, float]:
    if not epsilon > 0:
        raise ValueError("Epsilon must be strictly positive.")

    p_value = max(
        norm.cdf((estimate-epsilon)/standard_error),
        norm.sf((estimate+epsilon)/standard_error)
    )
    inverted = max(
        abs(estimate - abs(norm.ppf(alpha)) * standard_error),
        abs(estimate + abs(norm.ppf(alpha)) * standard_error)
    )
    return p_value, inverted


def equivalence(
        estimate: float,
        standard_error: float,
        epsilon: float,
        alpha: float = 0.05,
        search_tolerance: float = 1e-3,
        max_search_grid: float = 3.0,
) -> typing.Tuple[float, float]:
    if not epsilon > 0:
        raise ValueError("Epsilon must be strictly positive.")

    # equation 3 (Hartman, 2021)
    test_statistic = lambda x: ncx2.cdf(
        estimate**2 / standard_error**2,
        df=1,
        nc=x**2 / standard_error**2,
    )
    p_value = test_statistic(epsilon)
    if ncx2.cdf(estimate**2 / standard_error**2, df=1, nc=0) < alpha:
        # if noncentrality parameter estimate is so close to zero that p < alpha, return NA
        inverted = np.nan
    else:
        inverted, result = brentq(
            f=lambda x: test_statistic(x) - alpha,
            a=1e-5,
            b=max(10.0, max_search_grid * abs(estimate)*standard_error),
            xtol=search_tolerance,
            full_output=True,
            disp=False,
        )
        if not result.converged:
            warnings.warn('Unable to find equivalence confidence interval.')
            inverted = np.nan
    return p_value, inverted


def equivalence_density(
        estimate_density_left: float,
        estimate_density_right: float,
        standard_error_left: float,
        standard_error_right: float,
        epsilon: float = 1.5,
        alpha: float = 0.05,
) -> typing.Tuple[float, float]:
    if not epsilon > 0:
        raise ValueError("Epsilon must be strictly positive.")

    # equation 5 (Hartman, 2021)
    t1 = lambda x: (
        estimate_density_left - estimate_density_right / x
     ) / np.sqrt(
        standard_error_left ** 2  + standard_error_right ** 2 / x ** 2
    )
    t2 = lambda x: (
        estimate_density_left - estimate_density_right * x
     ) / np.sqrt(
        standard_error_left ** 2  + standard_error_right ** 2 * x ** 2
    )
    # calculate p-value
    p_value = max(norm.sf(t1(epsilon)), norm.cdf(t2(epsilon)))
    # construct equivalence confidence interval
    inverted = np.nan
    try:
        inverted, result = brentq(
            f=lambda x: max(norm.sf(t1(x)), norm.cdf(t2(x))) - alpha,
            a=1e-5,
            b=100,
            xtol=1e-4,
            full_output=True,
            disp=False,
        )
        success = result.converged
    except ValueError:
        success = False

    if not success:
        warnings.warn('Unable to find equivalence confidence interval.')

    return p_value, inverted