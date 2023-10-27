import random, time
from .feature import Feature
import numpy as np

class Tabular(object):
    def __init__(self, features: list[Feature]=[]) -> None:
        self._features = features
        pass

    def __getitem__(self, key) -> Feature:
        num_features = len(self._features)
        if ( isinstance(key, int) ):
            assert (key < num_features)
        elif ( isinstance(key, slice) ):
            assert (key.stop < num_features)

        return self._features[key]

    def __setitem__(self, key: int, value: Feature) -> Feature:
        assert (key < len(self._features))
        self._features[key] = value
        return self._features[key]

    def __len__(self) -> int:
        return len(self._features)

    def concat(self, rhs: Feature):
        self._features.append(rhs)
        return self

    def rand(self, num_examples: int, num_inputs: int=1, mean: np.float64=1.0, target_min: int=0, target_max: int=1):
        self._features = []
        random.seed(int(time.time()))
        for idx in range(num_examples):
            random_input = np.random.rand(num_inputs) * mean
            random_target = np.float64(random.randrange(target_min, target_max))
            self.concat(Feature(random_input, random_target))

        return self

    def __repr__(self) -> str:
        result = "[\n"
        for feature in self._features:
            result += f"  {feature}\n"
        result += "]"
        return result
