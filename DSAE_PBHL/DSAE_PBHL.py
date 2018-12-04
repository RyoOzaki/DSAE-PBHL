import tensorflow as tf
from DSAE_PBHL import DSAE_Soft, SAE, SAE_PBHL
from DSAE_PBHL.util import merge_dict

class DSAE_PBHL_Soft(DSAE_Soft):
    _a_network_class = SAE_PBHL

    def __init__(self, structure, structure_pb, **kwargs):
        self._structure_pb = structure_pb
        super(DSAE_PBHL_Soft, self).__init__(structure, **kwargs)

    def _stack_network(self, soft_kwargs=None, hard_kwargs=None):
        if soft_kwargs is None:
            soft_kwargs = {}
        if hard_kwargs is None:
            hard_kwargs = {}
        structure = self._structure
        structure_pb = self._structure_pb
        networks = self._networks = []
        L = len(structure)

        i = 0
        with tf.variable_scope(f"{i+1}_th_network"):
            net = self._a_soft_network_class(structure[i], structure[i+1], **soft_kwargs)
        networks.append(net)
        hidden_layer = net.hidden_layer
        for i in range(1, L-2):
            with tf.variable_scope(f"{i+1}_th_network"):
                net = self._a_soft_network_class(structure[i], structure[i+1], input_layer=hidden_layer, **soft_kwargs)
            hidden_layer = net.hidden_layer
            networks.append(net)
        i = L - 2
        with tf.variable_scope(f"{i+1}_th_network"):
            net = self._a_network_class(structure[i], structure[i+1], structure_pb[0], structure_pb[1], input_layer=hidden_layer, **hard_kwargs)
        networks.append(net)

    @property
    def input_layer_pb(self):
        return self._networks[-1].input_layer_pb

    def fit(self, sess, target_network_id, input, input_pb, epoch, extended_feed_dict=None, **kwargs):
        feed_dict = merge_dict({self.input_layer_pb: input_pb}, extended_feed_dict)
        return super(DSAE_PBHL_Soft, self).fit(sess, target_network_id, input, epoch, extended_feed_dict=feed_dict, **kwargs)

class DSAE_PBHL(DSAE_PBHL_Soft):
    _a_soft_network_class = SAE
    _a_network_class = SAE_PBHL
