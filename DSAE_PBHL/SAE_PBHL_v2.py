import tensorflow as tf
from .Model import PB_Model
from .SAE_PBHL import SAE_PBHL
from .util import merge_dict

class SAE_PBHL_v2(SAE_PBHL, PB_Model):
    """
    SAE_PBHL_v2: Sparce auto-encoder with parametic bias in hidden layer version 2
    """

    def _stack_network(self):
        input_dim       = self.input_dim
        input_dim_feat  = input_dim - self.input_dim_pb
        input_dim_pb    = self.input_dim_pb
        hidden_dim      = self.hidden_dim
        hidden_dim_feat = hidden_dim - self.hidden_dim_pb
        hidden_dim_pb   = self.hidden_dim_pb

        weight_initializer = self._weight_initializer
        bias_initializer   = self._bias_initializer

        encoder_activator = self._encoder_activator
        decoder_activator = self._decoder_activator
        activator_pb      = self._activator_pb

        with tf.variable_scope("parameters"):
            with tf.variable_scope("encoder"):
                encoder_weight_AB = tf.get_variable("encoder_weight_AB", shape=(input_dim_feat, hidden_dim), initializer=weight_initializer)
                encoder_weight_OO = tf.zeros((input_dim_pb, hidden_dim), name="encoder_weight_O")
                encoder_weight = tf.concat((encoder_weight_AB, encoder_weight_OO), axis=0, name="encoder_weight")
                encoder_bias = tf.get_variable("encoder_bias", shape=(hidden_dim, ), initializer=bias_initializer)
            with tf.variable_scope("decoder"):
                decoder_weight_X = tf.get_variable("decoder_weight_X", shape=(hidden_dim_feat, input_dim_feat), initializer=weight_initializer)
                decoder_weight_O = tf.zeros((hidden_dim_feat, input_dim_pb), name="decoder_weight_O")
                decoder_weight_XO = tf.concat((decoder_weight_X, decoder_weight_O), axis=1, name="decoder_weight_XO")
                decoder_weight_YZ = tf.get_variable("decoder_weight_YZ", shape=(hidden_dim_pb, input_dim), initializer=weight_initializer)
                decoder_weight = tf.concat((decoder_weight_XO, decoder_weight_YZ), axis=0, name="deocder_weight")
                decoder_bias = tf.get_variable("decoder_bias", shape=(input_dim, ), initializer=bias_initializer)

        with tf.variable_scope("layers"):
            input_layer       = self._input_layer
            with tf.variable_scope("hidde_layer"):
                hidden_layer      = encoder_activator(tf.matmul(input_layer, encoder_weight) + encoder_bias)
                hidden_layer      = tf.identity(hidden_layer, "hidden_layer")
            with tf.variable_scope("restoration_layer"):
                restoration_layer_unactivate = tf.matmul(hidden_layer, decoder_weight) + decoder_bias
                with tf.variable_scope("feature"):
                    restoration_layer_feature    = decoder_activator(restoration_layer_unactivate[:, :input_dim_feat])
                    restoration_layer_feature    = tf.identity(restoration_layer_feature, "restoration_layer_feature")
                with tf.variable_scope("parametric_bias"):
                    restoration_layer_pb         = activator_pb(restoration_layer_unactivate[:, input_dim_feat:])
                    restoration_layer_pb         = tf.identity(restoration_layer_pb, "restoration_layer_pb")
                restoration_layer = tf.concat((restoration_layer_feature, restoration_layer_pb), axis=1, name="restoration_layer")

        self._encoder_weight = encoder_weight
        self._encoder_bias   = encoder_bias
        self._decoder_weight = decoder_weight
        self._decoder_bias   = decoder_bias

        self._input_layer  = input_layer
        self._hidden_layer = hidden_layer
        self._hidden_layer_feature = tf.identity(hidden_layer[:, :hidden_dim_feat], "hidden_layer_feature")
        self._hidden_layer_pb      = tf.identity(hidden_layer[:, hidden_dim_feat:], "hidden_layer_pb")
        self._restoration_layer_feature = restoration_layer_feature
        self._restoration_layer_pb      = restoration_layer_pb
        self._restoration_layer = restoration_layer

        self._trainable_variables = [encoder_weight_AB, encoder_bias, decoder_weight_X, decoder_weight_YZ, decoder_bias]
        self._parameters = {"encoder_weight": encoder_weight, "encoder_bias": encoder_bias, "decoder_weight": decoder_weight, "decoder_bias": decoder_bias}
