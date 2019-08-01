from typing import Union

import numpy as np
from numpy import ndarray
from scipy.interpolate import Rbf

from interpolators.base import BaseInterpolator


class RbfInterpolator(BaseInterpolator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rbfi = Rbf(*self.points.T, self.values, function="linear")

    def interpolate(self, alpha, v, mcode, gamma, wt, wp) -> Union[ndarray, float]:
        if not alpha.shape:
            return self.rbfi(alpha, v, mcode, gamma, wt, wp)
        else:
            size = alpha.shape[0]
            results = np.zeros_like(alpha)
            for x in range(alpha.shape[0]):
                for y in range(alpha.shape[0]):
                    results[size - 1 - x][size - 1 - y] = self.rbfi(alpha[0][x], v[y][0], mcode, gamma, wt, wp)
            return results
