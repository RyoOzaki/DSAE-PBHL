import tensorflow as tf
from .util import merge_dict
from .Model import Model, PB_Model

class Deep_Model(object):

    def __init__(
        self,
        structure,
        classes,
        network_kwargs
        ):
        self._global_step = 0
        self._structure = structure
        self._L = len(structure)

        self._stack_network(classes, network_kwargs)

    def _stack_network(self, classes, network_kwargs):
        structure = self._structure
        L = self._L
        networks = self._networks = []

        with tf.variable_scope("1_th_network"):
            net = classes[0](structure[0], structure[1], **network_kwargs[0])
        networks.append(net)
        hidden_layer = net.hidden_layer
        for i in range(1, L-1):
            with tf.variable_scope(f"{i+1}_th_network"):
                net = classes[i](structure[i], structure[i+1], input_layer=hidden_layer, **network_kwargs[i])
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

    def fit(self, sess, target_network_id, input, epoch, extended_feed_dict=None, summary_writer=None):
        target_network = self.networks[target_network_id]
        feed_dict = merge_dict({self.input_layer: input}, extended_feed_dict)
        train_operator = target_network.train_operator
        for _ in range(epoch):
            sess.run(train_operator, feed_dict=feed_dict)
        step, loss, summary  = sess.run([target_network.local_step, target_network.loss, target_network.summary], feed_dict=feed_dict)
        if summary_writer is not None:
            summary_writer.add_summary(summary, step)
        return step, loss, summary


class Deep_PB_Model(Deep_Model):

    def __init__(
        self,
        structure,
        pb_structure,
        classes,
        network_kwargs
        ):
        self._pb_structure = pb_structure

        super(Deep_PB_Model, self).__init__(structure, classes, network_kwargs)

    def _stack_network(self, classes, network_kwargs):
        structure = self._structure
        pb_structure = self._pb_structure
        L = self._L
        networks = self._networks = []

        with tf.variable_scope("1_th_network"):
            if issubclass(classes[0], PB_Model):
                net = classes[0](structure[0], structure[1], pb_structure[0], pb_structure[1], **network_kwargs[0])
                self._input_layer_pb = net.input_layer_pb
            else:
                net = classes[0](structure[0], structure[1], **network_kwargs[0])
        networks.append(net)
        hidden_layer = net.hidden_layer
        for i in range(1, L-1):
            with tf.variable_scope(f"{i+1}_th_network"):
                if issubclass(classes[i], PB_Model):
                    net = classes[i](structure[i], structure[i+1], pb_structure[0], pb_structure[0], input_layer=hidden_layer, **network_kwargs[i])
                    self._input_layer_pb = net.input_layer_pb
                else:
                    net = classes[i](structure[i], structure[i+1], input_layer=hidden_layer, **network_kwargs[i])
            hidden_layer = net.hidden_layer
            networks.append(net)

    @property
    def input_layer_pb(self):
        return self._input_layer_pb

    def fit(self, sess, target_network_id, input, input_pb, epoch, extended_feed_dict=None, **kwargs):
        feed_dict = merge_dict({self.input_layer_pb: input_pb}, extended_feed_dict)
        return super(Deep_PB_Model, self).fit(sess, target_network_id, input, epoch, extended_feed_dict=feed_dict, **kwargs)
