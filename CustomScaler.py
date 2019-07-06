from typing import List

import numpy as np


class CustomScaler:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, data: np.ndarray) -> None:
        self.means = np.mean(data, 0)
        self.stds = np.std(data, 0)
        # print(self.means)
        # print(self.stds)

    def _check_fitted(self):
        if (self.means is None) or (self.stds is None):
            raise Exception("you need to first fit data")

    def transform_data(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (data - self.means) / self.stds
        # return data

    def transform_parameters(self, parameters: List) -> List:
        self._check_fitted()
        if len(parameters) != len(self.means):
            raise ValueError("incorrect number of parameters")
        for index, parameter in enumerate(parameters):
            yield (parameter - self.means[index]) / self.stds[index]
