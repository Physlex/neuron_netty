import numpy as np
from models.tabular import Tabular
from models.linear import LinearModel

def squared_diff(tabular_data: Tabular, linear_models: list[LinearModel]) -> np.float64:
    loss = np.float64(0.0)
    for idx in range(len(tabular_data)):
        loss += (tabular_data[idx].target - linear_models[idx].target) ** 2

    return loss
