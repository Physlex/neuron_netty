from data_streams import Tabular
import numpy as np


def gradient_descent(examples: np.ndarray, learning_rate: np.float64) -> np.ndarray:
    pass


def unit_tests() -> bool:
    ## FEATURE TESTS

    ## TABULAR TESTS
    example = Tabular().rand(num_examples=100, num_inputs=4, mean=22.456, target_max=1000)

    return True


def main():
    assert (unit_tests())

    pass


if __name__ == "__main__":
    main()
    pass
