from .Deep_Model import Deep_Model
from .AE import AE
from DSAE_PBHL.util import Builder

class DAE(Deep_Model):
    """
    DAE: Deep auto-encoder
    """
    _a_network_class = AE

    def __init__(self, structure, **kwargs):
        builder = Builder(structure[0])
        for node in structure[1:]:
            builder.stack(self._a_network_class, node, **kwargs)
        super(DAE, self).__init__(*builder.get_build_args())
