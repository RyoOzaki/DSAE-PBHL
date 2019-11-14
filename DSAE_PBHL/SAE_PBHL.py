import tensorflow as tf
from .Model import PB_Model
from .SAE import SAE
from .util import merge_dict

class SAE_PBHL(SAE, PB_Model):
    """
    SAE_PBHL: Sparce auto-encoder with parametic bias in hidden layer
    """
    def __init__(self,
        input_dim_feat, hidden_dim_feat,
        input_dim_pb, hidden_dim_pb,
        activator_pb=tf.nn.softmax,
        input_layer=None,
        **kwargs
        ):

        self._input_dim_pb  = input_dim_pb
        self._hidden_dim_pb = hidden_dim_pb
        self._activator_pb  = activator_pb
        self._input_layer_feature = input_layer or tf.placeholder(tf.float32, [None, input_dim_feat], name="input_layer_feature")

        self._input_layer_pb = tf.placeholder(tf.float32, [None, input_dim_pb], name="input_layer_pb")
        self._input_layer = tf.concat((self._input_layer_feature, self._input_layer_pb), axis=1)
        super(SAE_PBHL, self).__init__(input_dim_feat + input_dim_pb, hidden_dim_feat + hidden_dim_pb, input_layer=self._input_layer, **kwargs)

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
                encoder_weight_O = tf.zeros((input_dim_pb, hidden_dim_feat), name="encoder_weight_O")
                encoder_weight_C = tf.get_variable("encoder_weight_C", shape=(input_dim_pb, hidden_dim_pb))
                encoder_weight_OC = tf.concat((encoder_weight_O, encoder_weight_C), axis=1, name="encoder_weight_OC")
                encoder_weight = tf.concat((encoder_weight_AB, encoder_weight_OC), axis=0, name="encoder_weight")
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

        self._trainable_variables = [encoder_weight_AB, encoder_weight_C, encoder_bias, decoder_weight_X, decoder_weight_YZ, decoder_bias]

    def _collect_summary(self):
        sup_list = super(SAE_PBHL, self)._collect_summary()
        rlf = tf.summary.histogram("restoration_layer_feature", self._restoration_layer_feature)
        rlp = tf.summary.histogram("restoration_layer_pb", self._restoration_layer_pb)
        sup_list.extend([rlf, rlp])
        return sup_list

    @property
    def feature_input_dim(self):
        return self.input_dim - self.input_dim_pb

    @property
    def feature_hidden_dim(self):
        return self.hidden_dim - self.hidden_dim_pb

    @property
    def input_dim_pb(self):
        return self._input_dim_pb

    @property
    def hidden_dim_pb(self):
        return self._hidden_dim_pb

    @property
    def input_layer(self):
        return self._input_layer_feature

    @property
    def input_layer_pb(self):
        return self._input_layer_pb

    @property
    def hidden_layer(self):
        return self._hidden_layer_feature

    @property
    def hidden_layer_pb(self):
        return self._hidden_layer_pb

    def fit(self, sess, input, input_pb, epoch, extended_feed_dict=None, **kwargs):
        feed_dict = merge_dict({self.input_layer_pb: input_pb}, extended_feed_dict)
        return super(SAE_PBHL, self).fit(sess, input, epoch, extended_feed_dict=feed_dict, **kwargs)
