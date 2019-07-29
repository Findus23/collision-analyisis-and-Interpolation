import numpy as np
from numpy import ndarray
from scipy.interpolate import Rbf

from interpolators.base import BaseInterpolator


class RbfInterpolator(BaseInterpolator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rbfi = Rbf(*self.points.T, self.values, function="linear")

    def interpolate(self, alpha, v, mcode, gamma, wt, wp) -> ndarray:
        results = np.zeros_like(alpha)
        # print(v[30][0])
        # exit()
        for x in range(alpha.shape[0]):
            for y in range(alpha.shape[0]):
                results[99 - x][99 - y] = self.rbfi(alpha[0][x], v[y][0], mcode, gamma, wt, wp)
                # print(alpha[0][x], v[y][0])
        # print(results)
        # print(self.rbfi(alpha[0][99], v[30][0], mcode, gamma, wt, wp))
        # print(self.rbfi(60, 2.2, mcode, gamma, wt, wp))

        # exit()
        return results
