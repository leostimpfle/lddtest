import abc
import numpy as np


class KernelWeights(abc.ABC):

    def __init__(
            self,
            x: np.typing.ArrayLike,
            center: float,
            bandwidth: float,
    ):
        self._distance = (np.array(x) - center) / bandwidth

    @property
    def distance(self) -> np.typing.NDArray[float]:
        return self._distance

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    def weights(self) -> np.typing.NDArray[float]:
        x = self._weights_raw
        x = np.where(
            np.abs(self.distance) > 1,
            0,
            x,
        )
        x = x / x.sum()
        return x


    @property
    @abc.abstractmethod
    def _weights_raw(self) -> np.typing.NDArray[float]:
        pass


class Triangular(KernelWeights):

    @property
    def name(self) -> str:
        return 'triangular'


    @property
    def _weights_raw(self) -> np.typing.NDArray[float]:
        return 1 - np.abs(self.distance)
