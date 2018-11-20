import numpy as np

class Normalizer(object):

    def __init__(self, max_v=None, min_v=None):
        if max_v is not None:
            max_v = np.array(max_v)
        if min_v is not None:
            min_v = np.array(min_v)
        self._params = {"max_v": max_v, "min_v": min_v}

    @property
    def max_v(self):
        return self._params["max_v"]

    @max_v.setter
    def max_v(self, v):
        self._params["max_v"] = np.array(v)

    @property
    def min_v(self):
        return self._params["min_v"]

    @min_v.setter
    def min_v(self, v):
        self._params["min_v"] = np.array(v)

    def normalize(self, data):
        if self.max_v is None:
            self.max_v = data.max(axis=0)
        if self.min_v is None:
            self.min_v = data.min(axis=0)
        normalized = 2.0 * ((data - self.min_v) / (self.max_v - self.min_v)) - 1.0
        return normalized

    def unnormalize(self, data):
        if self.max_v is None or self.min_v is None:
            raise RuntimeError("Parameters are not initialized.")
        max_v = self.max_v
        min_v = self.min_v
        unnormalized = (max_v - min_v) * (data + 1.0) / 2.0 + min_v
        return unnormalized

    def save_params(self, f):
        np.savez(f, **self._params)

    def load_params(self, f):
        params = np.load(f)
        if "max_v" not in params or "min_v" not in params:
            raise RuntimeError("Parameters are not initialized.")
        self._params = params

    @classmethod
    def load(cls, f):
        norm = Normalizer()
        norm.load_params(f)
        return norm
