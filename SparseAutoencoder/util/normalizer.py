import numpy as np

class Normalizer(object):

    def __init__(self, max_v=None, min_v=None):
        if max_v is not None:
            max_v = np.array(max_v)
        if min_v is not None:
            min_v = np.array(min_v)
        if max_v is not None:
            dim = len(max_v)
        else:
            dim = None
        self.params = {"max_v": max_v, "min_v": min_v, "dim": dim}]

    @property
    def dim(self):
        return self.params["dim"]

    @property
    def max_v(self):
        return self.params["max_v"]

    @max_v.setter
    def max_v(self, v):
        if self.dim is None:
            self.params["max_v"] = np.array(v)
            self.params["dim"] = self.params["max_v"].shape[0]
        elif self.dim == len(v):
            self.params["max_v"] = np.array(v)
        else:
            raise ValueError()

    @property
    def min_v(self):
        return self.params["min_v"]

    @min_v.setter
    def min_v(self, v):
        if self.dim is None:
            self.params["min_v"] = np.array(v)
            self.params["dim"] = self.params["min_v"].shape[0]
        elif self.dim == len(v):
            self.params["min_v"] = np.array(v)
        else:
            raise ValueError()

    def normalize(self, data):
        if data.shape[1] != self.dim:
            ValueError()
        if self.max_v is None:
            max_v = self.max_v = data.max(axis=0)
        if self.min_v is None:
            min_v = self.min_v = data.min(axis=0)
        normalized = 2.0 * ((data - min_v) / (max_v - min_v)) - 1.0
        return normalized

    def save_params(self, p):
        np.savez(f, **self.params)

    def load_params(self, f):
        param = np.load(f)
        if param["dim"] is not None:
            if param["dim"] == self.dim:
                self.params = param
            else:
                raise ValueError()
