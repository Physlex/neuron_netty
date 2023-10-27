from .model import ModelBase
import numpy as np

class Feature(ModelBase):
    def __init__(self, input_features: np.ndarray, target_feature: np.float64) -> None:
        super().__init__(input_features, target_feature)
        pass

    def __mul__(self, rhs):
        for idx in range(len(self._domain)):
            self._domain[idx] *= rhs
        return self

    def __add__(self, rhs):
        for idx in range(len(self._domain)):
            self._domain[idx] += rhs
        return self

    def __sub__(self, rhs):
        for idx in range(len(self._domain)):
            self._domain[idx] -= rhs
        return self

    def __div__(self, rhs):
        for idx in range(len(self._domain)):
            self._domain[idx] /= rhs
        return self