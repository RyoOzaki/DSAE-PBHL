import tensorflow as tf
from .Model import Model
from .util import merge_dict

class AE(Model):
    """
    AE: Auto-encoder
    """
    def __init__(self,
        input_dim, hidden_dim,
        input_layer=None,
        activator=tf.nn.tanh,
        encoder_activator=None,
        decoder_activator=None,
        weight_initializer=tf.initializers.truncated_normal,
        bias_initializer=tf.initializers.truncated_normal
        ):
        assert input_dim > 0 and hidden_dim > 0
        assert activator is not None or (encoder_activator is not None and decoder_activator is not None)

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer

        if encoder_activator is None:
            encoder_activator = activator
        if decoder_activator is None:
            decoder_activator = activator
        self._encoder_activator = encoder_activator
        self._decoder_activator = decoder_activator

        if input_layer is None:
            self._input_layer = tf.placeholder(tf.float32, [None, input_dim], name="input_layer")
        else:
            self._input_layer = input_layer
        self._stack_network()
        self._define_loss()
        self._summary = tf.summary.merge(self._collect_summary())
        self._train_operator = self._get_train_operator(tf.train.AdamOptimizer(name="optimizer"))

    def _stack_network(self):
        input_dim  = self.input_dim
        hidden_dim = self.hidden_dim

        weight_initializer = self._weight_initializer
        bias_initializer   = self._bias_initializer

        encoder_activator = self._encoder_activator
        decoder_activator = self._decoder_activator

        with tf.variable_scope("parameters"):
            encoder_weight = tf.get_variable("encoder_weight", shape=(input_dim, hidden_dim), initializer=weight_initializer)
            encoder_bias   = tf.get_variable("encoder_bias", shape=(hidden_dim, ), initializer=bias_initializer)
            decoder_weight = tf.get_variable("decoder_weight", shape=(hidden_dim, input_dim), initializer=weight_initializer)
            decoder_bias   = tf.get_variable("decoder_bias", shape=(input_dim, ), initializer=bias_initializer)

        with tf.variable_scope("layers"):
            input_layer       = self._input_layer
            hidden_layer      = encoder_activator(tf.matmul(input_layer, encoder_weight) + encoder_bias)
            hidden_layer      = tf.identity(hidden_layer, "hidden_layer")
            restoration_layer = decoder_activator(tf.matmul(hidden_layer, decoder_weight) + decoder_bias)
            restoration_layer = tf.identity(restoration_layer, "restoration_layer")

        self._encoder_weight = encoder_weight
        self._encoder_bias   = encoder_bias
        self._decoder_weight = decoder_weight
        self._decoder_bias   = decoder_bias

        self._input_layer       = input_layer
        self._hidden_layer      = hidden_layer
        self._restoration_layer = restoration_layer

        self._trainable_variables = [encoder_weight, encoder_bias, decoder_weight, decoder_bias]

    def _define_loss(self):
        with tf.variable_scope("losses"):
            restoration_loss = self._get_restoration_loss()
            self._loss = restoration_loss

    def _collect_summary(self):
        ew = tf.summary.histogram("encoder_weight", self._encoder_weight)
        eb = tf.summary.histogram("encoder_bias", self._encoder_bias)
        dw = tf.summary.histogram("decoder_weight", self._decoder_weight)
        db = tf.summary.histogram("decoder_bias", self._decoder_bias)
        hl = tf.summary.histogram("hidden_layer", self._hidden_layer)
        rl = tf.summary.histogram("restoration_layer", self._restoration_layer)
        lo = tf.summary.scalar("total_loss", self._loss)
        return [ew, eb, dw, db, hl, rl, lo]

    def _get_restoration_loss(self):
        input  = self._input_layer
        output = self._restoration_layer
        loss = tf.reduce_mean((input - output)**2) / 2.0
        loss = tf.identity(loss, "restoration_loss")
        return loss

    def _get_train_operator(self, optimizer, **kwargs):
        loss = self.loss
        with tf.variable_scope("train_operator"):
            self._local_step = tf.get_variable("local_step", shape=(), trainable=False, dtype=tf.int32, initializer=tf.constant_initializer(0))
            train_variables = self.trainable_variables
            train_operator = optimizer.minimize(loss, var_list=train_variables, global_step=self._local_step, name="train_operator", **kwargs)
        return train_operator

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def output_dim(self):
        return self.input_dim

    @property
    def input_layer(self):
        return self._input_layer

    @property
    def hidden_layer(self):
        return self._hidden_layer

    @property
    def restoration_layer(self):
        return self._restoration_layer

    @property
    def trainable_variables(self):
        return self._trainable_variables

    @property
    def encoder_weight(self):
        return self._encoder_weight

    @property
    def encoder_bias(self):
        return self._encoder_bias

    @property
    def decoder_weight(self):
        return self._decoder_weight

    @property
    def decoder_bias(self):
        return self._decoder_bias

    @property
    def summary(self):
        return self._summary

    @property
    def loss(self):
        return self._loss

    @property
    def local_step(self):
        return self._local_step

    @property
    def train_operator(self):
        return self._train_operator

    def fit(self, sess, input, epoch, extended_feed_dict=None, summary_writer=None):
        feed_dict = merge_dict({self.input_layer: input}, extended_feed_dict)
        train_operator = self.train_operator
        for _ in range(epoch):
            sess.run(train_operator, feed_dict=feed_dict)
        result = sess.run([self.local_step, self.loss, self.summary], feed_dict=feed_dict)
        if summary_writer is not None:
            summary_writer.add_summary(result[2], result[0])
        return result
