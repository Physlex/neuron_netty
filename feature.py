import numpy as np

class Feature(object):
    def __init__(self, inputs: np.ndarray, target: np.float64) -> None:
        self._domain = []
        for idx in range(len(inputs)):
            self._domain.append(np.float64(inputs[idx]))

        self._range = np.float64(target)
        assert ( isinstance(self._range, np.float64) )

        pass

    @property
    def target(self):
        return self._range
    
    @property
    def inputs(self):
        return self._domain

    def __getitem__(self, key: int) -> np.float64:
        assert (key < len(self._domain))
        return tuple(self._domain[key])

    def __setitem__(self, key: int, value: np.float64) -> np.float64:
        assert (key < len(self._domain))
        self._domain[key] = value
        return value

    def __len__(self) -> int:
        return len(self._domain)

    def __repr__(self) -> str:
        func_repr: str = "( "
        for idx_x in range(len(self._domain)):
            func_repr += f"X{idx_x}: {self._domain[idx_x]}, "
        func_repr += f"Y0: {self._range} ) "
        return func_repr

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