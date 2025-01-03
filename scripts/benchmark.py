import itertools
from timeit import default_timer as timer
from datetime import timedelta

import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects

from lddtest.enums import Language

import lddtest
from lddtest.utils import sample_data
from tests.test_against_r import _none_to_null

if __name__ == '__main__':
    eqtesting = importr("eqtesting")
    cutoff = 0.0
    bandwidth = None
    epsilon = 1.5
    alpha = 0.05
    number_observations = [
        1_000,
        10_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
    ]
    number_clusters = [
        2,
        10,
        100
    ]
    runtimes = {}
    for ni, nc in itertools.product(number_observations, number_clusters):
        print(ni, nc)
        running, clusters = sample_data(
            number_observations=ni,
            number_clusters=nc,
        )
        start = timer()
        lddtest.lddtest(
            running=running,
            epsilon=epsilon,
            bandwidth=bandwidth,
            cutoff=cutoff,
            alpha=alpha,
            clusters=clusters,
        )
        end = timer()
        runtimes.update({(Language.python, ni, nc): timedelta(seconds=end-start)})

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

        start = timer()
        eqtesting.lddtest(
            runvar='rvar',
            data=data_r,
            cutpoint=cutoff,
            epsilon=epsilon,
            alpha=alpha,
            bw=_none_to_null(bandwidth),
            cluster='cvar' if clusters is not None else '',
            plot=False,
        )
        end = timer()
        runtimes.update({(Language.r, ni, nc): timedelta(seconds=end - start)})


