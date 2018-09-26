import numpy as np
import tensorflow as tf

#-------------------------------------------------------------------------------

def weight_variable(shape, **kwargs):
    # initial = tf.random_normal(shape, stddev=0.1)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, **kwargs)

def bias_variable(shape, **kwargs):
    # initial = tf.random_normal(shape, stddev=0.1)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, **kwargs)

class SAE(object):

    def __init__(self, n_in, n_hidden, alpha=0.003, beta=0.7, eta=0.5):
        self.params = {"input_dim": n_in, "hidden_dim": n_hidden,
            "alpha": alpha, "beta": beta, "eta": eta,
            "encode_W": None, "encode_b": None,
            "decode_W": None, "decode_b": None
            }
        self._define_network(n_in, n_hidden)
        self._define_loss(alpha, beta, eta)

    def _define_network(self, n_in, n_hidden):
        # Define network-----------
        self.input_layer = tf.placeholder(tf.float32, [None, n_in])

        self.enc_weight = weight_variable([n_in, n_hidden], trainable=True, name="encode_W")
        self.enc_bias = bias_variable([n_hidden], trainable=True, name="encode_b")

        self.hidden_layer = tf.tanh(tf.matmul(self.input_layer, self.enc_weight) + self.enc_bias)

        self.dec_weight = weight_variable([n_hidden, n_in], trainable=True, name="decode_W")
        self.dec_bias = bias_variable([n_in], trainable=True, name="decode_b")

        self.restoration_layer = tf.tanh(tf.matmul(self.hidden_layer, self.dec_weight) + self.dec_bias)

    def _define_loss(self, alpha, beta, eta):
        # Define loss function-----------
        hidden_layer_reduced = (1.0 + tf.reduce_mean(self.hidden_layer, axis=0)) / 2.0

        self.restoration_loss    = tf.reduce_mean(tf.pow(self.restoration_layer - self.input_layer, 2)) / 2.0
        # self.regularization_loss = (tf.reduce_sum(tf.pow(self.enc_weight, 2)) + tf.reduce_sum(tf.pow(self.dec_weight, 2))) / 2.0
        self.regularization_loss = tf.nn.l2_loss(self.enc_weight) + tf.nn.l2_loss(self.enc_weight)
        self.kl_divergence_loss  = tf.reduce_sum(
                                    eta * tf.log(eta) -
                                    eta * tf.log(hidden_layer_reduced) +
                                    (1.0 - eta) * tf.log(1.0 - eta) -
                                    (1.0 - eta) * tf.log(1.0 - hidden_layer_reduced)
                                    )
        self.loss = self.restoration_loss + alpha * self.regularization_loss + beta * self.kl_divergence_loss

    def encode(self, x_in):
        return np.tanh(np.dot(x_in, self.params["encode_W"]) + self.params["encode_b"])

    def decode(self, h_in):
        return np.tanh(np.dot(h_in, self.params["decode_W"]) + self.params["decode_b"])

    def fit(self, x_train, epoch=5, epsilon=0.000001):
        with tf.Session() as sess:
            optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            sess.run(tf.global_variables_initializer())
            last_loss = sess.run(self.loss, feed_dict={self.input_layer: x_train})
            t = 0
            print('\nStep: {}'.format(t))
            print("loss: {}".format(last_loss))
            while True:
                for _ in range(epoch):
                    sess.run(optimizer, feed_dict={self.input_layer: x_train})
                t += epoch
                loss = sess.run(self.loss, feed_dict={self.input_layer: x_train})
                print('\nStep: {}'.format(t))
                print("loss: {}".format(loss))
                if abs(last_loss - loss) < epsilon:
                    break
                last_loss = loss
            self.params["encode_W"] = sess.run(self.enc_weight)
            self.params["decode_W"] = sess.run(self.dec_weight)
            self.params["encode_b"] = sess.run(self.enc_bias)
            self.params["decode_b"] = sess.run(self.dec_bias)
            self.hiddens = sess.run(self.hidden_layer, feed_dict={self.input_layer: x_train})

    def save_params(self, f):
        np.savez(f, **self.params)

    def load_params(self, f):
        params = np.load(f)
        assert params["input_dim"] == self.params["input_dim"]
        assert params["hidden_dim"] == self.params["hidden_dim"]
        self.params = params

    @classmethod
    def load(cls, f):
        params = np.load(f)
        if "input_dim" not in params or "hidden_dim" not in params:
            raise RuntimeError("Does not have 'input_dim' or 'hidden_dim' or both.")
        sae = SAE(params["input_dim"], params["hidden_dim"])
        sae.load_params(f)
        return sae
