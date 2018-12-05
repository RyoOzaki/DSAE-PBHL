from .Deep_Model import Deep_Model
from .AE import AE
from .SAE import SAE
from .DAE import DAE
from DSAE_PBHL.util import Builder

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

        builder = Builder(structure[0])
        for node in structure[1:-1]:
            builder.stack(self._a_base_network_class, node, **kwargs_dict)
        builder.stack(self._a_final_network_class, structure[-1], **final_kwargs_dict)
        super(DSAE_Soft, self).__init__(*builder.get_build_args())
