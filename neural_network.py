import numpy as np
import loss as ls
from tabular import Tabular
from model import LinearModel

def gradient_descent(examples: np.ndarray, learning_rate: np.float64) -> np.ndarray:
    pass


def unit_tests() -> bool:
    ## FEATURE TESTS

    ## TABULAR TESTS
    examples = Tabular().rand(num_examples=10, num_inputs=4, mean=22.456, target_max=1000)
    models = []

    for idx in range(len(examples)):
        feature = examples[idx]
        weights = np.random.rand(len(feature))
        models.append(LinearModel(feature, weights))

    examples_loss = ls.squared_diff(tabular_data=examples, linear_models=models)
    print(examples, examples_loss)

    return True


def main():
    assert (unit_tests())

    pass


if __name__ == "__main__":
    main()
    pass
