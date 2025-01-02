import numpy as np
import plotly.express as px

import lddtest
from lddtest.utils import sample_data

if __name__ == '__main__':
    seed = 42
    N = 1_00_000
    running = sample_data(
        number_observations=N,
        seed=seed,
    )
    cutoff = 0.0
    bin_size = None
    bandwidth = 0.5

    px.histogram(running)
    result = lddtest.dcdensity(
        running=running,
        cutoff=cutoff,
        bin_size=bin_size,
        bandwidth=bandwidth,
    )
    print(result)