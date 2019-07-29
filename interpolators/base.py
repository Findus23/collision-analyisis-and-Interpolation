from numpy import ndarray


class BaseInterpolator():

    def __init__(self, points, values):
        self.points = points
        self.values = values

    def interpolate(self, alpha, v, mcode, gamma, wt, wp) -> ndarray:
        pass
