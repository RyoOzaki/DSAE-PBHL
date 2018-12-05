from .Deep_Model import Deep_Model
from .AE import AE
from .SAE import SAE
from .DAE import DAE

class DSAE(DAE):
    _a_network_class = SAE

class DSAE_Soft(Deep_Model):
    _a_base_network_class = AE
    _a_final_network_class = SAE

    def __init__(self, structure, kwargs_dict=None, final_kwargs_dict=None):
        if kwargs_dict is None:
            kwargs_dict = {}
        if final_kwargs_dict is None:
            final_kwargs_dict = {}
        L = len(structure)
        classes = [self._a_base_network_class, ] * (L - 2)
        classes.append(self._a_final_network_class)
        network_kwargs = [kwargs_dict, ] * (L - 2)
        network_kwargs.append(final_kwargs_dict)
        super(DSAE_Soft, self).__init__(structure, classes, network_kwargs)
