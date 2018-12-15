import numpy as np
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

    def hidden_layers_with_eval(self, sess, input, extended_feed_dict=None):
        feed_dict = merge_dict({self.input_layer: input}, extended_feed_dict)
        return sess.run(self.hidden_layers, feed_dict=feed_dict)

    @property
    def input_layer(self):
        return self.networks[0].input_layer

    @property
    def losses(self):
        return [net.loss for net in self.networks]

    def losses_with_eval(self, sess, input, extended_feed_dict=None):
        feed_dict = merge_dict({self.input_layer: input}, extended_feed_dict)
        return sess.run(self.losses, feed_dict=feed_dict)

    def fit(self, sess, target_network_id, input, epoch, extended_feed_dict=None, summary_writer=None):
        target_network = self.networks[target_network_id]
        feed_dict = merge_dict({self.input_layer: input}, extended_feed_dict)
        train_operator = target_network.train_operator
        epoch_range = range(epoch)
        for _ in epoch_range:
            sess.run(train_operator, feed_dict=feed_dict)
        loss, step, summary  = sess.run([target_network.loss, target_network.local_step, target_network.summary], feed_dict=feed_dict)
        if summary_writer is not None:
            summary_writer.add_summary(summary, step)
        return loss, step, summary

    def fit_with_cross(self, sess, target_network_id, input, cross_input, epoch, extended_feed_dict=None, cross_extended_feed_dict=None, summary_writer=None, cross_summary_writer=None):
        target_network = self.networks[target_network_id]
        feed_dict = merge_dict({self.input_layer: input}, extended_feed_dict)
        cross_feed_dict = merge_dict({self.input_layer: cross_input}, cross_extended_feed_dict)
        train_operator = target_network.train_operator
        epoch_range = range(epoch)
        for _ in epoch_range:
            sess.run(train_operator, feed_dict=feed_dict)
        loss = sess.run(target_network.loss, feed_dict=feed_dict)
        cross_loss = sess.run(target_network.loss, feed_dict=cross_feed_dict)
        if summary_writer is not None:
            step, summary  = sess.run([target_network.local_step, target_network.summary], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
        if cross_summary_writer is not None:
            step, summary  = sess.run([target_network.local_step, target_network.summary], feed_dict=cross_feed_dict)
            cross_summary_writer.add_summary(summary, step)
        return loss, cross_loss

    def fit_until(self, sess, target_network_id, input, epoch, epsilon, extended_feed_dict=None, summary_writer=None, ckpt_file=None, global_step=None):
        if global_step is None:
            global_step = 0
        feed_dict = merge_dict({self.input_layer: input}, extended_feed_dict)
        if ckpt_file is not None:
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        target_network = self.networks[target_network_id]
        loss = target_network.loss
        train_ope = target_network.train_operator
        last_loss = sess.run(loss, feed_dict=feed_dict)
        epsilon *= epoch
        epoch_range = range(epoch)
        while True:
            for _ in epoch_range:
                sess.run(train_ope, feed_dict=feed_dict)
            global_step += epoch
            new_loss, local_step, summary = sess.run([loss, target_network.local_step, target_network.summary], feed_dict=feed_dict)
            if summary_writer is not None:
                summary_writer.add_summary(summary, local_step)
            if abs(new_loss - last_loss) < epsilon:
                break
            last_loss = new_loss
        return global_step

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
                self._hidden_layer_pb = net.hidden_layer_pb
            else:
                net = classes[0](structure[0], structure[1], **network_kwargs[0])
        networks.append(net)
        hidden_layer = net.hidden_layer
        for i in range(1, L-1):
            with tf.variable_scope(f"{i+1}_th_network"):
                if issubclass(classes[i], PB_Model):
                    net = classes[i](structure[i], structure[i+1], pb_structure[0], pb_structure[1], input_layer=hidden_layer, **network_kwargs[i])
                    self._input_layer_pb = net.input_layer_pb
                    self._hidden_layer_pb = net.hidden_layer_pb
                else:
                    net = classes[i](structure[i], structure[i+1], input_layer=hidden_layer, **network_kwargs[i])
            hidden_layer = net.hidden_layer
            networks.append(net)

    @property
    def input_layer_pb(self):
        return self._input_layer_pb

    @property
    def hidden_layer_pb(self):
        return self._hidden_layer_pb

    def hidden_layers_with_eval(self, sess, input, input_pb=None, extended_feed_dict=None):
        if input_pb is None:
            input_pb = np.zeros((input.shape[0], self._pb_structure[0]))
        feed_dict = merge_dict({self.input_layer_pb: input_pb}, extended_feed_dict)
        return super(Deep_PB_Model, self).hidden_layers_with_eval(sess, input, extended_feed_dict=feed_dict)

    def hidden_layer_pb_with_eval(self, sess, input, input_pb, extended_feed_dict=None):
        feed_dict = merge_dict({self.input_layer: input, self.input_layer_pb: input_pb}, extended_feed_dict)
        return sess.run(self.hidden_layer_pb, feed_dict=feed_dict)

    def losses_with_eval(self, sess, input, input_pb, extended_feed_dict=None):
        feed_dict = merge_dict({self.input_layer_pb: input_pb}, extended_feed_dict)
        return super(Deep_PB_Model, self).losses_with_eval(sess, input, extended_feed_dict=feed_dict)

    def fit(self, sess, target_network_id, input, input_pb, epoch, extended_feed_dict=None, **kwargs):
        feed_dict = merge_dict({self.input_layer_pb: input_pb}, extended_feed_dict)
        return super(Deep_PB_Model, self).fit(sess, target_network_id, input, epoch, extended_feed_dict=feed_dict, **kwargs)

    def fit_with_cross(self, sess, target_network_id, input, cross_input, input_pb, cross_input_pb, epoch, extended_feed_dict=None, cross_extended_feed_dict=None, **kwargs):
        feed_dict = merge_dict({self.input_layer_pb: input_pb}, extended_feed_dict)
        cross_feed_dict = merge_dict({self.input_layer_pb: cross_input_pb}, cross_extended_feed_dict)
        return super(Deep_PB_Model, self).fit_with_cross(sess, target_network_id, input, cross_input, epoch, extended_feed_dict=feed_dict, cross_extended_feed_dict=cross_feed_dict, **kwargs)

    def fit_until(self, sess, target_network_id, input, input_pb, epoch, epsilon, extended_feed_dict=None, **kwargs):
        feed_dict = merge_dict({self.input_layer_pb: input_pb}, extended_feed_dict)
        return super(Deep_PB_Model, self).fit_until(sess, target_network_id, input, epoch, epsilon, extended_feed_dict=feed_dict, **kwargs)
