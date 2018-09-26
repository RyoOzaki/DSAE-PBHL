import numpy as np
import tensorflow as tf
from util.tf_variables import bias_variable, weight_variable


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

    @property
    def encode_weight(self):
        return self.params["encode_W"]

    @property
    def encode_bias(self):
        return self.params["encode_b"]

    @property
    def decode_weight(self):
        return self.params["decode_W"]

    @property
    def decode_bias(self):
        return self.params["decode_b"]


    def encode(self, x_in):
        return np.tanh(np.dot(x_in, self.params["encode_W"]) + self.params["encode_b"])

    def decode(self, h_in):
        return np.tanh(np.dot(h_in, self.params["decode_W"]) + self.params["decode_b"])

    def feature(self, x_in):
        return self.encode(x_in)

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
        self.load_params_by_dict(self, params)

    def load_params_by_dict(self, dic):
        if "input_dim" not in dic or "hidden_dim" not in dic:
            raise RuntimeError("Does not have 'input_dim' or 'hidden_dim' or both.")
        self.params = dic

    @classmethod
    def load(cls, source):
        if type(source) is dict:
            params = source
        else:
            params = np.load(source)
        if "input_dim" not in params or "hidden_dim" not in params:
            raise RuntimeError("Does not have 'input_dim' or 'hidden_dim' or both.")
        instance = cls(params["input_dim"], params["hidden_dim"])
        instance.load_params_by_dict(params)
        return instance

class SAE_PBHL(SAE):

    def _define_network(self, n_in, n_hidden):
        # -----------------------------
        # v  = [x p]
        # h  = [z s]
        # W  = [[A B]
        #       [O C]]
        # W' = [[X O]
        #       [Y Z]]
        #
        # h  = v W  + b
        # v' = h W' + b'
        # Define network-----------
        self.input_layer = tf.placeholder(tf.float32, [None, sum(n_in)])

        enc_weight_A = weight_variable([n_in[0], n_hidden[0]], trainable=True, name="encode_W_A")
        enc_weight_B = weight_variable([n_in[0], n_hidden[1]], trainable=True, name="encode_W_B")
        enc_weight_O =        tf.zeros([n_in[1], n_hidden[0]])
        enc_weight_C = weight_variable([n_in[0], n_hidden[1]], trainable=True, name="encode_W_C")
        enc_weight_AB = tf.concat([enc_weight_A, enc_weight_B], axis=1)
        enc_weight_OC = tf.concat([enc_weight_O, enc_weight_C], axis=1)
        self.enc_weight = tf.concat([enc_weight_AB, enc_weight_OC], axis=0)
        self.enc_bias = bias_variable([sum(n_hidden)], trainable=True, name="encode_b")

        self.hidden_layer = tf.tanh(tf.matmul(self.input_layer, self.enc_weight) + self.enc_bias)

        dec_weight_X = weight_variable([n_hidden[0], n_in[0]], trainable=True, name="decode_W_X")
        dec_weight_O =        tf.zeros([n_hidden[0], n_in[1]])
        dec_weight_Y = weight_variable([n_hidden[1], n_in[0]], trainable=True, name="decode_W_Y")
        dec_weight_Z = weight_variable([n_hidden[0], n_in[1]], trainable=True, name="decode_W_Z")
        dec_weight_XO = tf.concat([dec_weight_X, dec_weight_O], axis=1)
        dec_weight_YZ = tf.concat([dec_weight_Y, dec_weight_Z], axis=1)
        self.dec_weight = tf.concat([dec_weight_XO, dec_weight_YZ], axis=0)
        self.dec_bias = bias_variable([sum(n_in)], trainable=True, name="decode_b")

        self.restoration_layer = tf.tanh(tf.matmul(self.hidden_layer, self.dec_weight) + self.dec_bias)

    def feature(self, x_in):
        return self.encode(x_in)[:, self.param["hidden_dim"][0]]
