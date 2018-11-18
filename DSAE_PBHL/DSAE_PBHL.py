import numpy as np
import tensorflow as tf
from DSAE_PBHL import DSAE

class DSAE_PBHL(DSAE):

    def __init__(self, structure, pb_structure, pb_activator=tf.nn.sigmoid, **kwargs):
        self.pb_activator = pb_activator
        self.pb_structure = pb_structure

        super(DSAE_PBHL, self).__init__(structure, **kwargs)

    def _stack_network(self):
        structure = self.structure
        pb_structure = self.pb_structure
        activator = self.activator
        pb_activator = self.pb_activator
        weight_initializer = self.weight_initializer
        bias_initializer = self.bias_initializer
        encode_layer = self._tf_encode_layer = []
        encode_weight = self._tf_encode_weight = []
        encode_bias = self._tf_encode_bias = []
        decode_layer = self._tf_decode_layer = []
        decode_weight = self._tf_decode_weight = []
        decode_bias = self._tf_decode_bias = []

        self._tf_input = tf.placeholder(tf.float32, [None, structure[0]], name="input")
        self._tf_pb_input = tf.placeholder(tf.float32, [None, pb_structure[0]], name="pb_input")
        encode_layer.append(self._tf_input)

        for i in range(len(structure) - 1):
            # stack network
            # structure[i] -> structure[i+1] -> structure[i]
            with tf.variable_scope(f"{i+1}_th_network", reuse=tf.AUTO_REUSE):

                if i == len(structure) - 2:
                    pb_input = self._tf_pb_input
                    input = tf.concat((encode_layer[-1], pb_input), axis=1)
                    encode_layer.append(input)

                    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                        with tf.variable_scope("parameters", reuse=tf.AUTO_REUSE):
                            enc_weight_AB = tf.get_variable("encoder_weight_AB", shape=(structure[i], structure[i+1]+pb_structure[1]), initializer=weight_initializer)
                            enc_weight_O = tf.zeros((pb_structure[0], structure[i+1]), name="encoder_weight_O")
                            enc_weight_C = tf.get_variable("encoder_weight_C", shape=(pb_structure[0], pb_structure[1]))
                            enc_weight_OC = tf.concat((enc_weight_O, enc_weight_C), axis=1, name="encoder_weight_OC")
                            enc_weight = tf.concat((enc_weight_AB, enc_weight_OC), axis=0, name="encoder_weight")
                            enc_bias = tf.get_variable("encoder_bias", shape=(structure[i+1]+pb_structure[1], ), initializer=bias_initializer)
                        with tf.variable_scope("hidden_layer", reuse=tf.AUTO_REUSE):
                            hidden = activator(tf.matmul(input, enc_weight) + enc_bias)
                            hidden = tf.identity(hidden, name="hidden_layer")

                    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                        with tf.variable_scope("parameters", reuse=tf.AUTO_REUSE):
                            dec_weight_X = tf.get_variable("decoder_weight_X", shape=(structure[i+1], structure[i]), initializer=weight_initializer)
                            dec_weight_O = tf.zeros((structure[i+1], pb_structure[0]), name="decoder_weight_O")
                            dec_weight_XO = tf.concat((dec_weight_X, dec_weight_O), axis=1, name="decoder_weight_XO")
                            dec_weight_YZ = tf.get_variable("decoder_weight_YZ", shape=(pb_structure[1], structure[i]+pb_structure[0]), initializer=weight_initializer)
                            dec_weight = tf.concat((dec_weight_XO, dec_weight_YZ), axis=0, name="deocder_weight")
                            dec_bias = tf.get_variable("decoder_bias", shape=(structure[i]+pb_structure[0], ), initializer=bias_initializer)
                        with tf.variable_scope("restoration_layer", reuse=tf.AUTO_REUSE):
                            unactivate_restoration = tf.matmul(hidden, dec_weight) + dec_bias
                            activated_feature = activator(unactivate_restoration[:, :structure[i]])
                            activated_pb = pb_activator(unactivate_restoration[:, structure[i]:])
                            restoration = tf.concat((activated_feature, activated_pb), axis=1, name="restoration_layer")

                        tf.summary.histogram("hidden_layer_feature", activated_feature)
                        tf.summary.histogram("hidden_layer_parametric_bias", activated_pb)

                else:

                    input = encode_layer[-1]
                    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                        with tf.variable_scope("parameters", reuse=tf.AUTO_REUSE):
                            enc_weight = tf.get_variable("encoder_weight", shape=(structure[i], structure[i+1]), initializer=weight_initializer)
                            enc_bias   = tf.get_variable("encoder_bias", shape=(structure[i+1], ), initializer=bias_initializer)
                        with tf.variable_scope("hidden_layer", reuse=tf.AUTO_REUSE):
                            hidden = activator(tf.matmul(input, enc_weight) + enc_bias)
                            hidden = tf.identity(hidden, name="hidden_layer")

                    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                        with tf.variable_scope("parameters", reuse=tf.AUTO_REUSE):
                            dec_weight = tf.get_variable("decoder_weight", shape=(structure[i+1], structure[i]), initializer=weight_initializer)
                            dec_bias   = tf.get_variable("decoder_bias", shape=(structure[i], ), initializer=bias_initializer)
                        with tf.variable_scope("restoration_layer", reuse=tf.AUTO_REUSE):
                            restoration = activator(tf.matmul(hidden, dec_weight) + dec_bias)
                            restoration = tf.identity(restoration, name="restoration_layer")

                tf.summary.histogram("hidden_layer", hidden)

            encode_weight.append(enc_weight)
            encode_bias.append(enc_bias)
            decode_weight.append(dec_weight)
            decode_bias.append(dec_bias)
            encode_layer.append(hidden)
            decode_layer.append(restoration)

        self._tf_hidden = hidden

    def _define_loss(self):
        structure = self.structure
        pb_structure = self.pb_structure
        encode_layer = self._tf_encode_layer
        encode_weight = self._tf_encode_weight
        decode_layer = self._tf_decode_layer
        decode_weight = self._tf_decode_weight

        restoration_loss = self._tf_restoration_loss = []
        regularization_loss = self._tf_regularization_loss = []
        kl_divergence_loss = self._tf_kl_divergence_loss = []
        total_loss = self._tf_loss = []

        with tf.variable_scope("hyper_parameters", reuse=tf.AUTO_REUSE):
            tf_alpha = tf.constant(self.alpha, name="alpha", dtype=tf.float32)
            tf_beta = tf.constant(self.beta, name="beta", dtype=tf.float32)
            tf_eta = tf.constant(self.eta, name="eta", dtype=tf.float32)

        for i in range(len(structure) - 1):
            # define loss at structure[i] -> structure[i+1] -> structure[i]
            if i == len(structure) - 2:
                input = encode_layer[i+1]
                hidden = encode_layer[i+2]
                output = decode_layer[i]
            else:
                input = encode_layer[i]
                hidden = encode_layer[i+1]
                output = decode_layer[i]

            with tf.variable_scope(f"{i+1}_th_network/", reuse=tf.AUTO_REUSE):
                with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):

                    with tf.variable_scope("restoration_loss", reuse=tf.AUTO_REUSE):
                        rest_loss = tf.reduce_mean(tf.pow(output - input, 2)) / 2.0
                        rest_loss = tf.identity(rest_loss, name="restoration_loss")

                    with tf.variable_scope("regularization_loss", reuse=tf.AUTO_REUSE):
                        regu_loss = tf.nn.l2_loss(encode_weight[i]) + tf.nn.l2_loss(decode_weight[i])
                        regu_loss = tf.identity(regu_loss, name="regularization_loss")

                    with tf.variable_scope("kl_divergence_loss", reuse=tf.AUTO_REUSE):
                        hidden_reduced = (1.0 + tf.reduce_mean(hidden, axis=0)) / 2.0
                        kl_loss = tf.reduce_sum(
                                tf_eta[i] * tf.log(tf_eta[i]) -
                                tf_eta[i] * tf.log(hidden_reduced) +
                                (1.0 - tf_eta[i]) * tf.log(1.0 - tf_eta[i]) -
                                (1.0 - tf_eta[i]) * tf.log(1.0 - hidden_reduced)
                                )
                        kl_loss = tf.identity(kl_loss, name="kl_divergence_loss")

                    loss = rest_loss + tf_alpha[i] * regu_loss + tf_beta[i] * kl_loss
                    loss = tf.identity(loss, name="total_loss")

                tf.summary.scalar("restoration_loss", rest_loss)
                tf.summary.scalar("regularization_loss", regu_loss)
                tf.summary.scalar("kl_divergence_loss", kl_loss)
                tf.summary.scalar("total_loss", loss)

            restoration_loss.append(rest_loss)
            regularization_loss.append(regu_loss)
            kl_divergence_loss.append(kl_loss)
            total_loss.append(loss)

    def fit(self, x_in, pb_in, **kwargs):
        extended_feed_dict = {self._tf_pb_input: pb_in}
        super(DSAE_PBHL, self).fit(x_in, extended_feed_dict=extended_feed_dict, **kwargs)

    def encode(self, x_in, pb_in):
        assert self._sess is not None, "Session is not initialized!! Please call initialize_variables or load_variables."
        sess = self._sess
        feed_dict = {self._tf_input: x_in, self._tf_pb_input: pb_in}
        return sess.run(self._tf_hidden, feed_dict=feed_dict)

    def feature(self, x_in):
        N = x_in.shape[0]
        dummy_pb_input = np.zeros((N, self.pb_structure[0]))
        return self.encode(x_in, dummy_pb_input)[:, :self.structure[-1]]
