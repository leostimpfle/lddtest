import numpy as np
from lddtest.utils import sample_data

def f1(generator, running, clusters, unique_clusters):
    drawn = generator.choice(
        unique_clusters,
        size=unique_clusters.shape[0],
        replace=True,
    )
    return np.concatenate([running[clusters == cluster] for cluster in drawn])

def f2(generator, running, clusters, unique_clusters):
    drawn = generator.choice(
        unique_clusters,
        size=unique_clusters.shape[0],
        replace=True,
    )
    tmp = np.where(
        # array of shape (number observations, number clusters)
        # true if i-th observation is in the j-th cluster
        clusters[:, None] == np.repeat(
            drawn[None, :],
            repeats=clusters.shape[0],
            axis=0,
        ),
        running[:, None],
        np.nan,  # set to nan if i-th observation is not in j-th cluster
    ).flatten()
    return tmp[np.isfinite(tmp)]

if __name__ == '__main__':
    seed = 1
    ni = 1_000_00
    nc = 1000
    running, clusters = sample_data(
        number_observations=ni,
        number_clusters=nc,
    )
    unique_clusters = np.unique(clusters)
    generator = np.random.default_rng(seed=seed)

