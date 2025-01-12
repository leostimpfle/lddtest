import math
import multiprocessing
import typing
import pandas as pd
import numpy as np
from scipy.stats import norm

from lddtest.dcdensity import dcdensity
from lddtest.enums import DcdensityResults, LddtestResults

# Fitzgerald, Jack 2024: Manipulation Tests in Regression Discontinuity Design: The Need for Equivalence Testing
# URL: https://www.econstor.eu/handle/10419/300277
# Code: https://github.com/jack-fitzgerald/eqtesting

def lddtest(
        running: np.typing.ArrayLike,
        epsilon: float,
        cutoff: float,
        alpha: float = 0.05,
        bin_size: typing.Optional[float] = None,
        bandwidth: typing.Optional[float] = None,
        clusters: typing.Optional[np.typing.ArrayLike] = None,
        bootstrap: bool = True,
        bootstrap_iterations: int = 1_000,
) -> pd.Series:
    output = {}
    # perform McCrary density test
    density_test, _, _ = dcdensity(
        running=running,
        cutoff=cutoff,
        bin_size=bin_size,
        bandwidth=bandwidth,
    )
    estimate = density_test[DcdensityResults.estimate]
    standard_error = density_test[DcdensityResults.standard_error]
    rope = np.array([-math.log(epsilon), math.log(epsilon)])
    # set bound closest to estimate
    index_closest_to_estimate = np.argmin(
        np.abs(rope - estimate),
        keepdims=False,
    )
    bound = rope[index_closest_to_estimate]
    # number of observations
    notnull = ~np.isnan(running)
    is_within_bandwidth = (
            running >= (cutoff - density_test[DcdensityResults.bandwidth])
    ) & (
            running <= (cutoff + density_test[DcdensityResults.bandwidth])
    )
    if clusters is not None:
        notnull &= ~np.isnan(clusters)

    # store in output
    output.update(
        {
            LddtestResults.estimate: estimate,
            LddtestResults.number_observations: notnull.sum(),
            LddtestResults.number_observations_effective: (
                    notnull & is_within_bandwidth
            ).sum(),
            LddtestResults.epsilon_lower: rope[0],
            LddtestResults.epsilon_upper: rope[1],
            LddtestResults.bandwidth: density_test[DcdensityResults.bandwidth],
        }
    )

    if not bootstrap and clusters is None:
        z_stat = (estimate - bound) / standard_error
        if index_closest_to_estimate == 0:
            p_value = norm.cdf(z_stat)
        else:
            p_value = norm.sf(z_stat)

        ci_left = estimate - norm.ppf(1 - alpha) * standard_error
        ci_right = estimate + norm.ppf(1 - alpha) * standard_error

    else:
        # (cluster) bootstrap inference
        estimates = _boostrap(
            running=running,
            dcdensity_results=density_test,
            clusters=clusters,
            max_iter=bootstrap_iterations,
        )
        standard_error = np.std(estimates)
        ci_left = np.quantile(
            estimates,
            q=alpha,
        )
        ci_right = np.quantile(
            estimates,
            q=1-alpha,
        )
        z_stat = estimate / standard_error  # TODO: is this correct?
        p_value = (
            (estimates >= rope[0]) & (estimates <= rope[1])
        ).sum() / bootstrap_iterations

    output.update(
        {
            LddtestResults.standard_error: standard_error,
            LddtestResults.confidence_lower_equivalence: ci_left,
            LddtestResults.confidence_upper_equivalence: ci_right,
            LddtestResults.z_stat_equivalence: z_stat,
            LddtestResults.p_value_equivalence: p_value,
        }
    )
    output = pd.Series(output, name='lddtest')
    output = output.loc[[n for n in LddtestResults]]  # sort output
    return output


def _boostrap(
        running: np.typing.ArrayLike,
        dcdensity_results: pd.Series,
        clusters: typing.Optional[np.typing.ArrayLike] = None,
        max_iter: int = 1_000,
        seed: int = 42,
        n_workers: typing.Optional[int] = -1,
) -> np.typing.NDArray[float]:
    unique_clusters = None if clusters is None else np.unique(clusters)
    if unique_clusters.shape[0] < 2:
        # nothing to cluster
        clusters = None

    if n_workers < 0:
        # use all available cores
        n_workers = multiprocessing.cpu_count()

    cutoff = dcdensity_results[DcdensityResults.cutoff]
    bin_size = dcdensity_results[DcdensityResults.bin_size]
    bandwidth = dcdensity_results[DcdensityResults.bandwidth]
    generator = np.random.default_rng(seed=seed)
    if n_workers is not None:
        # multiprocessing
        args = [
            (generator, running, cutoff, bin_size, bandwidth, clusters, unique_clusters)
            for _ in range(max_iter)
        ]
        with multiprocessing.Pool(n_workers) as pool:
            estimates = pool.starmap(
                func=_iteration,
                iterable=args,
            )
    else:
        iteration = 0
        estimates = []
        while iteration < max_iter:
            iteration += 1
            result = _iteration(
                generator=generator,
                running=running,
                cutoff=cutoff,
                bin_size=bin_size,
                bandwidth=bandwidth,
                clusters=clusters,
                unique_clusters=unique_clusters,
            )
            estimates.append(result)

    estimates = np.array(
        [
            tmp[DcdensityResults.estimate] for tmp in estimates
        ]
    )
    return estimates


def _iteration(
        generator: np.random._generator.Generator,
        running: np.typing.ArrayLike,
        cutoff: float,
        bin_size: float,
        bandwidth: float,
        clusters: typing.Optional[np.typing.ArrayLike] = None,
        unique_clusters: typing.Optional[np.typing.ArrayLike] = None,
):
    random_sample = _sample(
        generator=generator,
        running=running,
        clusters=clusters,
        unique_clusters=unique_clusters,
    )
    is_permissible = (
         cutoff > random_sample.min()
    ) & (
        cutoff < random_sample.max()
    )
    if not is_permissible:
        # skip because sample does not include cutoff
        return

    result, _, _ = dcdensity(
        running=random_sample,
        cutoff=cutoff,
        bin_size=bin_size,
        bandwidth=bandwidth,
    )
    return result


def _sample(
        generator: np.random._generator.Generator,
        running: np.typing.ArrayLike,
        clusters: typing.Optional[np.typing.ArrayLike] = None,
        unique_clusters: typing.Optional[np.typing.ArrayLike] = None,
) -> np.typing.NDArray[float]:
    if clusters is None:
        tmp = generator.choice(
            running,
            size=running.shape[0],
            replace=True,
        )
    elif unique_clusters is not None:
        # draw clusters
        clusters_sampled = generator.choice(
            unique_clusters,
            size=unique_clusters.shape[0],
            replace=True,
        )
        # create clustered sample
        tmp = np.concatenate(
            [
                running[clusters == cluster] for cluster in clusters_sampled
            ]
        )
    else:
        raise ValueError(
            'Need to provide unique clusters'
        )

    return tmp




