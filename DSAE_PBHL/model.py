import numpy as np
import tensorflow as tf
from .util.tf_variables import bias_variable, weight_variable

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class SAE(object):

    def __init__(self, n_in, n_hidden, alpha=0.003, beta=0.7, eta=0.5):
        alpha = float(alpha)
        beta  = float(beta)
        eta   = float(eta)
        self._params = {"input_dim": n_in, "hidden_dim": n_hidden,
            "alpha": alpha, "beta": beta, "eta": eta,
            "encode_W": None, "encode_b": None,
            "decode_W": None, "decode_b": None
            }
        self._define_network(n_in, n_hidden)
        self._define_loss(alpha, beta, eta)

    def _define_network(self, n_in, n_hidden):
        # Define network-----------
        self._tf_input_layer = tf.placeholder(tf.float32, [None, n_in])

        self._tf_enc_weight = weight_variable([n_in, n_hidden], trainable=True, name="encode_W")
        self._tf_enc_bias = bias_variable([n_hidden], trainable=True, name="encode_b")

        self._tf_hidden_layer = tf.tanh(tf.matmul(self._tf_input_layer, self._tf_enc_weight) + self._tf_enc_bias)

        self._tf_dec_weight = weight_variable([n_hidden, n_in], trainable=True, name="decode_W")
        self._tf_dec_bias = bias_variable([n_in], trainable=True, name="decode_b")

        self._tf_restoration_layer = tf.tanh(tf.matmul(self._tf_hidden_layer, self._tf_dec_weight) + self._tf_dec_bias)

    def _define_loss(self, alpha, beta, eta):
        # Define loss function-----------
        hidden_layer_reduced = (1.0 + tf.reduce_mean(self._tf_hidden_layer, axis=0)) / 2.0

        self._tf_restoration_loss    = tf.reduce_mean(tf.pow(self._tf_restoration_layer - self._tf_input_layer, 2)) / 2.0
        self._tf_regularization_loss = tf.nn.l2_loss(self._tf_enc_weight) + tf.nn.l2_loss(self._tf_dec_weight)
        self._tf_kl_divergence_loss  = tf.reduce_sum(
                                    eta * tf.log(eta) -
                                    eta * tf.log(hidden_layer_reduced) +
                                    (1.0 - eta) * tf.log(1.0 - eta) -
                                    (1.0 - eta) * tf.log(1.0 - hidden_layer_reduced)
                                    )
        self._tf_loss = self._tf_restoration_loss + alpha * self._tf_regularization_loss + beta * self._tf_kl_divergence_loss

    @property
    def input_dim(self):
        return self._params["input_dim"]

    @property
    def hidden_dim(self):
        return self._params["hidden_dim"]

    @property
    def encode_weight(self):
        return self._params["encode_W"]

    @property
    def encode_bias(self):
        return self._params["encode_b"]

    @property
    def decode_weight(self):
        return self._params["decode_W"]

    @property
    def decode_bias(self):
        return self._params["decode_b"]

    def encode(self, x_in):
        return np.tanh(np.dot(x_in, self._params["encode_W"]) + self._params["encode_b"])

    def decode(self, h_in):
        return np.tanh(np.dot(h_in, self._params["decode_W"]) + self._params["decode_b"])

    def feature(self, x_in):
        return self.encode(x_in)

    def fit(self, x_in, epoch=5, epsilon=0.000001, print_loss=True):
        with tf.Session() as sess:
            optimizer = tf.train.AdamOptimizer().minimize(self._tf_loss)
            sess.run(tf.global_variables_initializer())
            last_loss = sess.run(self._tf_loss, feed_dict={self._tf_input_layer: x_in})
            t = 0
            if print_loss:
                print('Step: {}'.format(t))
                print("loss: {}".format(last_loss))
            while True:
                for _ in range(epoch):
                    sess.run(optimizer, feed_dict={self._tf_input_layer: x_in})
                t += epoch
                loss = sess.run(self._tf_loss, feed_dict={self._tf_input_layer: x_in})
                if abs(last_loss - loss) < epsilon:
                    break
                elif print_loss:
                    print('Step: {}'.format(t))
                    print("Loss: {}".format(loss))
                    print()
                last_loss = loss
            self._params["encode_W"] = sess.run(self._tf_enc_weight)
            self._params["decode_W"] = sess.run(self._tf_dec_weight)
            self._params["encode_b"] = sess.run(self._tf_enc_bias)
            self._params["decode_b"] = sess.run(self._tf_dec_bias)
        print('Total Step: {}'.format(t))
        print("Final Loss: {}".format(loss))

    def save_params(self, f):
        np.savez(f, **self._params)

    def load_params(self, f):
        params = np.load(f)
        self.load_params_by_dict(self, params)

    def load_params_by_dict(self, dic):
        assert "input_dim" in dic
        assert "hidden_dim" in dic
        # if "input_dim" not in dic or "hidden_dim" not in dic:
        #     raise RuntimeError("Does not have 'input_dim' or 'hidden_dim' or both.")
        self._params = dic

    @classmethod
    def load(cls, source):
        if type(source) is dict:
            params = source
        else:
            params = np.load(source)
        assert "input_dim" in params
        assert "hidden_dim" in params
        # if "input_dim" not in params or "hidden_dim" not in params:
        #     raise RuntimeError("Does not have 'input_dim' or 'hidden_dim' or both.")
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
        # h  = tanh(v W  + b)
        # v' = [x' p']
        # x' = tanh(zX + sY + b'_1)
        # p' = sigmoid(sZ + b'_2)
        # Define network-----------
        self._tf_input_layer = tf.placeholder(tf.float32, [None, sum(n_in)])

        enc_weight_A  = weight_variable([n_in[0], n_hidden[0]], trainable=True, name="encode_W_A")
        enc_weight_B  = weight_variable([n_in[0], n_hidden[1]], trainable=True, name="encode_W_B")
        enc_weight_O  =        tf.zeros([n_in[1], n_hidden[0]])
        enc_weight_C  = weight_variable([n_in[1], n_hidden[1]], trainable=True, name="encode_W_C")
        enc_weight_AB = tf.concat([enc_weight_A, enc_weight_B], axis=1)
        enc_weight_OC = tf.concat([enc_weight_O, enc_weight_C], axis=1)
        self._tf_enc_weight = tf.concat([enc_weight_AB, enc_weight_OC], axis=0)
        self._tf_enc_bias   = bias_variable([sum(n_hidden)], trainable=True, name="encode_b")

        self._tf_hidden_layer = tf.tanh(tf.matmul(self._tf_input_layer, self._tf_enc_weight) + self._tf_enc_bias)

        dec_weight_X  = weight_variable([n_hidden[0], n_in[0]], trainable=True, name="decode_W_X")
        dec_weight_O  =        tf.zeros([n_hidden[0], n_in[1]])
        dec_weight_Y  = weight_variable([n_hidden[1], n_in[0]], trainable=True, name="decode_W_Y")
        dec_weight_Z  = weight_variable([n_hidden[1], n_in[1]], trainable=True, name="decode_W_Z")
        dec_weight_XO = tf.concat([dec_weight_X, dec_weight_O], axis=1)
        dec_weight_YZ = tf.concat([dec_weight_Y, dec_weight_Z], axis=1)
        self._tf_dec_weight = tf.concat([dec_weight_XO, dec_weight_YZ], axis=0)
        self._tf_dec_bias   = bias_variable([sum(n_in)], trainable=True, name="decode_b")

        self._tf_restoration_layer = tf.tanh(tf.matmul(self._tf_hidden_layer, self._tf_dec_weight) + self._tf_dec_bias)
        # If we use the other activation function in parametric bias, we will use the code as follows.
        # restoration_unactivated = tf.matmul(self._tf_hidden_layer, self._tf_dec_weight) + self._tf_dec_bias
        # restoration_feature     = tf.tanh(restoration_unactivated[:, :n_in[0]])
        # restoration_pb          = tf.sigmoid(restoration_unactivated[:, n_in[0]:])
        # # restoration_pb          = tf.nn.softmax(restoration_unactivated[:, n_in[0]:])
        # self._tf_restoration_layer = tf.concat([restoration_feature, restoration_pb], axis=1)

    def fit(self, x_in, x_pb, **kwargs):
        x_train = np.concatenate([x_in, x_pb], axis=1)
        super(SAE_PBHL, self).fit(x_train, **kwargs)

    def encode(self, x_in, x_pb):
        return super(SAE_PBHL, self).encode(np.concatenate([x_in, x_pb], axis=1))

    def decode(self, h_concat):
        # unactivated = np.dot(h_concat, self._params["decode_W"]) + self._params["decode_b"]
        # col = self._params["input_dim"][0]
        # rest_in = np.tanh(unactivated[:, :col])
        # rest_pb = sigmoid(unactivated[:, col:])
        # return rest_in, rest_pb
        activated = np.tanh(np.dot(h_concat, self._params["decode_W"]) + self._params["decode_b"])
        col = self._params["input_dim"][0]
        return activated[:,:col], activated[:, col:]

    def decode_pb(self, h_pb):
        row = self._params["hidden_dim"][0]
        col = self._params["input_dim"][0]
        decode_W = self._params["decode_W"][row:, col:]
        decode_b = self._params["decode_b"][col:]
        # return sigmoid(np.dot(h_pb, decode_W) + decode_b)
        return np.tanh(np.dot(h_pb, decode_W) + decode_b)

    def decode_feature(self, h_in, h_pb):
        return self.decode(np.concatenate([h_in, h_pb], axis=1))[0]

    def feature(self, x_in):
        row = self._params["input_dim"][0]
        col = self._params["hidden_dim"][0]
        encode_W = self._params["encode_W"][:row, :col]
        encode_b = self._params["encode_b"][:col]
        return np.tanh(np.dot(x_in, encode_W) + encode_b)

    def feature_pb(self, x_in, x_pb):
        x_concat = np.concatenate([x_in, x_pb], axis=1)
        col = self._params["hidden_dim"][0]
        encode_W = self._params["encode_W"][:, col:]
        encode_b = self._params["encode_b"][col:]
        return np.tanh(np.dot(x_concat, encode_W) + encode_b)
