import numpy as np
import tensorflow as tf

class DSAE(object):

    def __init__(self, structure, alpha=0.003, beta=0.7, eta=0.5,
        activator=tf.nn.tanh,
        weight_initializer=tf.initializers.truncated_normal,
        bias_initializer=tf.initializers.truncated_normal
        ):
        assert structure is not None and len(structure) >= 2, "DSAE structure must be having the least 1 hidden layer."
        self._sess = None
        self.structure = structure
        self.activator = activator
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        alpha = np.array(alpha)
        beta = np.array(beta)
        eta = np.array(eta)
        L = len(structure)
        if alpha.ndim == 0:
            alpha = np.ones(L-1) * alpha
        if beta.ndim == 0:
            beta = np.ones(L-1) * beta
        if eta.ndim == 0:
            eta = np.ones(L-1) * eta
        assert alpha.ndim == 1 and alpha.shape[0] == L-1
        assert beta.ndim == 1 and beta.shape[0] == L-1
        assert eta.ndim == 1 and eta.shape[0] == L-1
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

        self._stack_network()
        # self._define_decode_network()
        self._define_loss()
        self._define_train_operator()
        self._define_summary()


    def _stack_network(self):
        structure = self.structure
        activator = self.activator
        weight_initializer = self.weight_initializer
        bias_initializer = self.bias_initializer
        encode_layer = self._tf_encode_layer = []
        encode_weight = self._tf_encode_weight = []
        encode_bias = self._tf_encode_bias = []
        decode_layer = self._tf_decode_layer = []
        decode_weight = self._tf_decode_weight = []
        decode_bias = self._tf_decode_bias = []

        self._tf_input = tf.placeholder(tf.float32, [None, structure[0]], name="input")
        encode_layer.append(self._tf_input)

        for i in range(len(structure) - 1):
            # stack network
            # structure[i] -> structure[i+1] -> structure[i]
            with tf.variable_scope(f"{i+1}_th_network", reuse=tf.AUTO_REUSE):

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

    def _define_train_operator(self):
        L = len(self.structure)
        train_operator = self._tf_train_operator = []
        loss = self._tf_loss

        for i in range(L-1):
            with tf.variable_scope(f"{i+1}_th_network/", reuse=tf.AUTO_REUSE):
                    target_loss = loss[i]
                    target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"{i+1}_th_network/")

                    optimizer = tf.train.AdamOptimizer(name="optimizer")
                    train_ope = optimizer.minimize(target_loss, var_list=target_variables, name="train_operator")
            train_operator.append(train_ope)

    def _define_summary(self):
        L = len(self.structure)
        self._tf_summary = []
        for i in range(L-1):
            summary = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=f"{i+1}_th_network/")
            self._tf_summary.append(tf.summary.merge(summary))

    def init_session(self):
        self.close_session()
        if self._sess is None:
            self.set_session(tf.Session())

    def set_session(self, sess):
        self._sess = sess

    def close_session(self):
        if self._sess is not None:
            self._sess.close()

    def initialize_variables(self):
        assert self._sess is not None, "Session is not initialized!! Please call init_session or set_session."
        initializer = tf.global_variables_initializer()
        self._sess.run(initializer)

    def load_variables(self, ckpt_dir=None, ckpt_file=None):
        assert self._sess is not None, "Session is not initialized!! Please call init_session or set_session."
        saver = tf.train.Saver()
        if ckpt_file is None:
            if ckpt_dir is None:
                ckpt_dir = './ckpt'
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            assert ckpt is not None, "ckpt file not found."
            ckpt_file = ckpt.model_checkpoint_path
        saver.restore(self._sess, ckpt_file)

    # train_network = 0 -> all network
    def fit(self, x_in, epoch=5, epsilon=0.000001, print_loss=True, train_network=0, ckpt_file=None, summary_prefix=None, extended_feed_dict=None):
        assert self._sess is not None, "Session is not initialized!! Please call init_session or set_session."
        sess = self._sess
        if ckpt_file is None:
            ckpt_file = "ckpt/model.ckpt"
        if summary_prefix is None:
            summary_prefix = "graph"
        saver = tf.train.Saver()
        feed_dict = {self._tf_input: x_in}
        if extended_feed_dict is not None:
            feed_dict = {**extended_feed_dict, **feed_dict}
        range_epoch = list(range(epoch))
        if train_network == 0:
            train_network = range(1, len(self.structure))
        elif type(train_network) == int:
            train_network = [train_network, ]

        global_step = 0
        for target_network in train_network:
            local_step = 0
            summary_writer = tf.summary.FileWriter(f"{summary_prefix}/{target_network}_th_network_train", sess.graph)
            tf_summary = self._tf_summary[target_network-1]
            train_operator = self._tf_train_operator[target_network-1]
            tf_loss = self._tf_loss[target_network-1]
            last_loss = sess.run(tf_loss, feed_dict=feed_dict)
            if print_loss:
                print(f"Training network: {target_network}")
                print(f"Global step: {global_step}")
                print(f'Local step : {local_step}')
                print(f"loss: {last_loss}")
                print()
            while True:
                for _ in range_epoch:
                    sess.run(train_operator, feed_dict=feed_dict)
                global_step += epoch
                local_step += epoch
                summary, loss = sess.run((tf_summary, tf_loss), feed_dict=feed_dict)
                summary_writer.add_summary(summary, local_step)
                saver.save(sess, ckpt_file, global_step=global_step)

                if abs(last_loss - loss) < epsilon:
                    break
                elif print_loss:
                    print(f"Training network: {target_network}")
                    print(f"Global step: {global_step}")
                    print(f'Local step : {local_step}')
                    print(f"Loss: {loss}")
                    print()
                last_loss = loss

    def encode(self, x_in):
        assert self._sess is not None, "Session is not initialized!! Please call initialize_variables or load_variables."
        sess = self._sess
        feed_dict = {self._tf_input: x_in}
        return sess.run(self._tf_hidden, feed_dict=feed_dict)
