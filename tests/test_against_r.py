import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, NULL

import lddtest
from lddtest.utils import sample_data
from lddtest.enums import Language

rdd = importr("rdd")

def _run_r_dcdensity(
        running: FloatVector,
        cutoff: float,
        bandwidth: float = None,
        ext: bool = True,
):
    result = rdd.DCdensity(
        runvar=running,
        cutpoint=cutoff,
        bw=bandwidth if bandwidth is not None else NULL,
        ext=ext,
    )
    keep_names = {
        'theta': 'density discontinuity (log difference)',
        'se': 'density discontinuity (standard error)',
        'z': 'z-statistic',
        'p': 'p-value',
        'binsize': 'bin size',
        'bw': 'bandwidth',
        'cutpoint': 'cutoff',
    }
    result = pd.Series(
        {
            keep_names[name]: result[i][0]
            for i, name in enumerate(result.names)
            if name in keep_names
        },
        name=Language.r.value,
    )
    return result


def test_dcdensity():
    data = sample_data()
    cutoff = 0.0
    bandwidth = None
    result_r = _run_r_dcdensity(
        running=FloatVector(data),
        cutoff=cutoff,
        bandwidth=bandwidth
    )
    result_python = lddtest.dcdensity(
        running=data,
        cutoff=cutoff,
        bandwidth=bandwidth,
    )
    result = result_python.to_frame(
        Language.python.value
    ).join(
        result_r,
        how='left',
        validate='one_to_one',
    )
    np.testing.assert_allclose(
        result[Language.python.value],
        result[Language.r.value],
        rtol=1e-5,
    )
