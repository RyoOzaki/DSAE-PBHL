from .DSAE_PBHL import DSAE_PBHL_Soft
from .AE import AE
from .SAE import SAE
from .SAE_PBHL_v2 import SAE_PBHL_v2

class DSAE_PBHL_v2_Soft(DSAE_PBHL_Soft):
    _a_base_network_class = AE
    _a_final_network_class = SAE_PBHL_v2

class DSAE_PBHL_v2(DSAE_PBHL_v2_Soft):
    _a_base_network_class = SAE
