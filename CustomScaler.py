from typing import List, Iterator

import numpy as np


class CustomScaler:
    """
    This is basically a simpler implementation of `sklearn.preprocessing.StandardScaler` that
    allows transforming both parameter sets and the initial data.
    """

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

    def transform_parameters(self, parameters: List) -> Iterator[float]:
        self._check_fitted()
        if len(parameters) != len(self.means):
            raise ValueError("incorrect number of parameters")
        for index, parameter in enumerate(parameters):
            yield (parameter - self.means[index]) / self.stds[index]
