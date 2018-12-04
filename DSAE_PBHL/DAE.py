import tensorflow as tf
from DSAE_PBHL import AE
from DSAE_PBHL.util import merge_dict

class DAE(object):
    """
    DAE: Deep auto-encoder
    """
    _a_network_class = AE

    def __init__(self, structure, **kwargs):
        self._structure = structure
        self._L = len(structure)

        self._stack_network(**kwargs)

    def _stack_network(self, **kwargs):
        structure = self._structure
        L = self._L
        networks = self._networks = []

        with tf.variable_scope("1_th_network"):
            net = self._a_network_class(structure[0], structure[1], **kwargs)
        networks.append(net)
        hidden_layer = net.hidden_layer
        for i in range(1, L-1):
            with tf.variable_scope(f"{i+1}_th_network"):
                net = self._a_network_class(structure[i], structure[i+1], input_layer=hidden_layer, **kwargs)
            hidden_layer = net.hidden_layer
            networks.append(net)

    @property
    def networks(self):
        return self._networks

    @property
    def hidden_layers(self):
        return [net.hidden_layer for net in self.networks]

    @property
    def input_layer(self):
        return self.networks[0].input_layer

    @property
    def losses(self):
        return [net.loss for net in self.networks]

    @property
    def trainable_variables_list(self):
        return [net.trainable_variables for net in self.networks]

    def fit(self, sess, target_network_id, input, epoch, extended_feed_dict=None, summary_writer=None, **kwargs):
        target_network = self.networks[target_network_id]
        feed_dict = merge_dict({self.input_layer: input}, extended_feed_dict)
        train_operator = target_network.train_operator
        for _ in range(epoch):
            sess.run(train_operator, feed_dict=feed_dict)
        result = sess.run([target_network.local_step, target_network.loss, target_network.summary], feed_dict=feed_dict)
        if summary_writer is not None:
            summary_writer.add_summary(result[2], result[0])
        return result
