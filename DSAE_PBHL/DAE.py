from .Deep_Model import Deep_Model
from .AE import AE

class DAE(Deep_Model):
    """
    DAE: Deep auto-encoder
    """
    _a_network_class = AE

    def __init__(self, structure, **kwargs):
        L = len(structure)
        classes = [self._a_network_class, ] * (L - 1)
        network_kwargs = [kwargs, ] * (L - 1)
        super(DAE, self).__init__(structure, classes, network_kwargs)
