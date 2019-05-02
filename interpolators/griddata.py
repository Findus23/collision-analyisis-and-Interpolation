from numpy import ndarray, save
from scipy.interpolate import griddata

from interpolators.base import BaseInterpolator


class GriddataInterpolator(BaseInterpolator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def interpolate(self, alpha, v, mcode, gamma, wt, wp) -> ndarray:
        return griddata(self.points, self.values, (alpha, v, mcode, gamma, wt, wp), method="linear")
