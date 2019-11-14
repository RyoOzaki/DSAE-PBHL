from .Deep_Model import Deep_PB_Model
from .AE import AE
from .SAE import SAE
from .SAE_PBHL import SAE_PBHL
from DSAE_PBHL.util import Builder

class DSAE_PBHL_Soft(Deep_PB_Model):
    _a_base_network_class = AE
    _a_final_network_class = SAE_PBHL

    def __init__(self, structure, pb_structure, kwargs_dict=None, final_kwargs_dict=None):
        kwargs_dict = kwargs_dict or {}
        final_kwargs_dict = final_kwargs_dict or {}

        builder = Builder(structure[0], pb_input_dim=pb_structure[0])
        for node in structure[1:-1]:
            builder.stack(self._a_base_network_class, node, **kwargs_dict)
        builder.stack(self._a_final_network_class, structure[-1], pb_hidden_dim=pb_structure[1], **final_kwargs_dict)
        super(DSAE_PBHL_Soft, self).__init__(*builder.get_build_args())


class DSAE_PBHL(DSAE_PBHL_Soft):
    _a_base_network_class = SAE
