import tensorflow as tf
from DSAE_PBHL import DAE, AE, SAE

class DSAE_Soft(DAE):
    _a_soft_network_class = AE
    _a_network_class = SAE

    def _stack_network(self, soft_kwargs=None, hard_kwargs=None):
        if soft_kwargs is None:
            soft_kwargs = {}
        if hard_kwargs is None:
            hard_kwargs = {}
        structure = self._structure
        networks = self._networks = []
        L = self._L

        i = 0
        with tf.variable_scope(f"{i+1}_th_network"):
            net = self._a_soft_network_class(structure[0], structure[1], **soft_kwargs)
        networks.append(net)
        hidden_layer = net.hidden_layer
        for i in range(1, L-2):
            with tf.variable_scope(f"{i+1}_th_network"):
                net = self._a_soft_network_class(structure[i], structure[i+1], input_layer=hidden_layer, **soft_kwargs)
            hidden_layer = net.hidden_layer
            networks.append(net)
        i = L - 2
        with tf.variable_scope(f"{i+1}_th_network"):
            net = self._a_network_class(structure[i], structure[i+1], input_layer=hidden_layer, **hard_kwargs)
        networks.append(net)

class DSAE(DAE):
    _a_network_class = SAE
