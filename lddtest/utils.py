import typing
import numpy as np


def round_to_integer(number: np.typing.NDArray[float]) -> np.typing.NDArray[int]:
    # Python's built-in `round` may round up or down if two integers are equally close: https://docs.python.org/3/library/functions.html#round
    # # Use decimal.ROUND_HALF_UP to enforce consistency
    # tmp = decimal.Decimal(number).quantize(
    #     decimal.Decimal(1),
    #     rounding=decimal.ROUND_HALF_UP,
    # )
    is_scalar = False
    if np.isscalar(number):
        is_scalar = True
        number = np.array(number)

    number = (1e-10 + number).round(decimals=0).astype(int)
    if is_scalar:
        number = number.squeeze().item()

    return number


def sample_data(
        number_observations: int = 100_000,
        seed: int = 42,
        number_clusters: int = 1,
) -> typing.Tuple[np.typing.NDArray[float], np.typing.NDArray[int]]:
    np.random.seed(seed)
    running = np.random.standard_normal(number_observations)
    clusters = np.random.randint(
        low=0,
        high=number_clusters,
        size=number_observations,
    )
    return running, clusters