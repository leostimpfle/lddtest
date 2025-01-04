import typing
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects
from rpy2.robjects import FloatVector, NULL, pandas2ri

import lddtest
from lddtest.utils import sample_data
from lddtest.enums import Language, LddtestResults, DcdensityResults


def _none_to_null(argument):
    if argument is None:
        return NULL
    else:
        return argument


def _run_r_dcdensity(
        running: np.typing.ArrayLike,
        cutoff: float,
        bandwidth: float = None,
        ext: bool = True,
) -> pd.Series:
    rdd = importr("rdd")
    result = rdd.DCdensity(
        runvar=FloatVector(running),
        cutpoint=cutoff,
        bw=_none_to_null(bandwidth),
        ext=ext,
    )
    keep_names = {
        'theta': DcdensityResults.estimate,
        'se': DcdensityResults.standard_error,
        'z': DcdensityResults.z_stat,
        'p': DcdensityResults.p_value,
        'binsize': DcdensityResults.bin_size,
        'bw': DcdensityResults.bandwidth,
        'cutpoint': DcdensityResults.cutoff,
    }
    result = pd.Series(
        {
            keep_names[name]: result[i][0]
            for i, name in enumerate(result.names)
            if name in keep_names
        },
        name=Language.r,
    )
    return result


def test_dcdensity(relative_tolerance: float = 1e-2):
    running, clusters = sample_data()
    cutoff = 0.0
    bandwidth = None
    result_r = _run_r_dcdensity(
        running=running,
        cutoff=cutoff,
        bandwidth=bandwidth,
    )
    result_python, _, _ = lddtest.dcdensity(
        running=running,
        cutoff=cutoff,
        bandwidth=bandwidth,
    )
    result = result_python.to_frame(
        Language.python
    ).join(
        result_r,
        how='left',
        validate='one_to_one',
    )
    np.testing.assert_allclose(
        result[Language.python],
        result[Language.r],
        rtol=relative_tolerance,
    )


def _run_r_lddtest(
        running: np.typing.ArrayLike,
        cutoff: float,
        epsilon: float,
        bandwidth: float = None,
        alpha: float = 0.05,
        clusters: typing.Optional[np.typing.ArrayLike] = None,
) -> pd.Series:
    eqtesting = importr("eqtesting")
    data = pd.concat(
        [
            pd.Series(running, name='rvar'),
            pd.Series(clusters, name='cvar'),
        ],
        axis=1,
    )
    data.dropna(how='all', axis=1, inplace=True)
    # convert pandas.DataFrame to R
    # https://rpy2.github.io/doc/latest/html/pandas.html#from-pandas-to-r
    with (rpy2.robjects.default_converter + pandas2ri.converter).context():
        data_r = rpy2.robjects.conversion.get_conversion().py2rpy(data)
    # rpy2.robjects.r.assign('data_r', data_r)
    output = eqtesting.lddtest(
        runvar='rvar',
        data=data_r,
        cutpoint=cutoff,
        epsilon=epsilon,
        alpha=alpha,
        bw=_none_to_null(bandwidth),
        cluster='cvar' if clusters is not None else '',
        plot=False,
    )
    keep_names = {
        'N': LddtestResults.number_observations,
        'Effective N': LddtestResults.number_observations_effective,
        'Epsilon Lower Bound': LddtestResults.epsilon_lower,
        'Epsilon Upper Bound': LddtestResults.epsilon_upper,
        'Theta': LddtestResults.estimate,
        'SE': LddtestResults.standard_error,
        'ECI Lower Bound': LddtestResults.confidence_lower_equivalence,
        'ECI Upper Bound': LddtestResults.confidence_upper_equivalence,
        'Equivalence z-statistic': LddtestResults.z_stat_equivalence,
        'p-value': LddtestResults.p_value_equivalence,
    }
    result = pd.Series(
        {
            keep_names[name]: str(output[0][i][0])
            for i, name in enumerate(output[0].names)
            if name in keep_names
        },
        name=Language.r,
    )
    result = pd.to_numeric(result, errors='coerce')
    result = result.loc[[n for n in LddtestResults if n in result.index]]
    return result


def test_lddtest(
        absolute_tolerance: float = 1e-1,
        relative_tolerance: float = 1e-2,
):
    number_observations = 10_000
    number_clusters = 10
    running, clusters = sample_data(
        number_observations=number_observations,
        number_clusters=number_clusters,
    )
    cutoff = 0.0
    bandwidth = None
    epsilon = 1.5
    alpha = 0.05
    result_r = _run_r_lddtest(
        running=running,
        clusters=clusters,
        cutoff=cutoff,
        bandwidth=bandwidth,
        epsilon=epsilon,
        alpha=alpha,
    )
    result_python = lddtest.lddtest(
        running=running,
        clusters=clusters,
        cutoff=cutoff,
        bandwidth=bandwidth,
        epsilon=epsilon,
        alpha=alpha,
    )
    result = result_python.to_frame(
        Language.python
    ).join(
        result_r,
        how='left',
        validate='one_to_one',
    )
    test_absolute = [
        LddtestResults.estimate,
        LddtestResults.standard_error,
        LddtestResults.confidence_lower_equivalence,
        LddtestResults.confidence_upper_equivalence,
        LddtestResults.epsilon_lower,
        LddtestResults.epsilon_upper,
    ]
    np.testing.assert_allclose(
        result.loc[result.index.isin(test_absolute), Language.python],
        result.loc[result.index.isin(test_absolute), Language.r],
        atol=absolute_tolerance,
    )
    test_relative = [
        LddtestResults.number_observations,
        LddtestResults.number_observations_effective,
    ]
    np.testing.assert_allclose(
        result.loc[result.index.isin(test_relative), Language.python],
        result.loc[result.index.isin(test_relative), Language.r],
        rtol=relative_tolerance,
    )