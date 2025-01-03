import numpy as np
import plotly.express as px
from scipy.stats import alpha

import lddtest
from lddtest.utils import sample_data

if __name__ == '__main__':
    seed = 42
    N = 1_000_000
    N_clusters = 1_000
    running, clusters = sample_data(
        number_observations=N,
        seed=seed,
        number_clusters=N_clusters,
    )
    cutoff = 0.0
    bin_size = None
    bandwidth = None
    # px.histogram(running)
    # result = lddtest.dcdensity(
    #     running=running,
    #     cutoff=cutoff,
    #     bin_size=bin_size,
    #     bandwidth=bandwidth,
    # )
    result = lddtest.lddtest(
        running=running,
        cutoff=cutoff,
        bin_size=bin_size,
        bandwidth=bandwidth,
        epsilon=1.5,
        alpha=0.05,
    )