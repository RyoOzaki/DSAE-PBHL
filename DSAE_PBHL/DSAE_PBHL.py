from .Deep_Model import Deep_PB_Model
from .AE import AE
from .SAE import SAE
from .SAE_PBHL import SAE_PBHL

class DSAE_PBHL_Soft(Deep_PB_Model):
    _a_base_network_class = AE
    _a_final_network_class = SAE_PBHL

    def __init__(self, structure, pb_structure, kwargs_dict=None, final_kwargs_dict=None):
        if kwargs_dict is None:
            kwargs_dict = {}
        if final_kwargs_dict is None:
            final_kwargs_dict = {}
        L = len(structure)
        classes = [self._a_base_network_class, ] * (L - 2)
        classes.append(self._a_final_network_class)
        network_kwargs = [kwargs_dict, ] * (L - 2)
        network_kwargs.append(final_kwargs_dict)
        super(DSAE_PBHL_Soft, self).__init__(structure, pb_structure, classes, network_kwargs)

class DSAE_PBHL(DSAE_PBHL_Soft):
    _a_base_network_class = SAE
