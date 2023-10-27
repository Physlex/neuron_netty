from .model import ModelBase
from .feature import Feature
import numpy as np

class LinearModel(ModelBase):
    def __init__(self, input_features: Feature, weights: list[np.float64]) -> None:
        input_features:np.ndarray = np.array(input_features.inputs)
        target_feature:np.ndarray = input_features.dot(weights)
        super().__init__(input_features, target_feature)
        pass
