from abc import ABC
from feature import Feature
import numpy as np

class ModelBase(ABC):
    def __init__(self, input_features: list[np.float64], target_feature) -> None:
        self._domain = []
        for feature in input_features:
            self._domain.append(np.float64(feature))
        self._range = np.float64(target_feature)
        assert ( isinstance(self._range, np.float64) )
        pass

    @property
    def target(self):
        return self._range

    def __repr__(self) -> str:
        result = "["
        for idx in range(len(self._domain)):
            result += f"X{idx}: {self._domain[idx]}, "
        result += f"Y0: {self.target}]"
        return result

class LinearModel(ModelBase):
    def __init__(self, input_features: Feature, weights: list[np.float64]) -> None:
        input_features:np.ndarray = np.array(input_features.inputs)
        target_feature:np.ndarray = input_features.dot(weights)
        super().__init__(input_features, target_feature)
        pass
