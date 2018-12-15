import tensorflow as tf
from .AE import AE

class SAE(AE):
    """
    SAE: Sparse auto-encoder
    """
    def __init__(self,
        input_dim, hidden_dim,
        alpha=0.003, beta=0.7, eta=0.5,
        **kwargs
        ):
        self._alpha = alpha
        self._beta  = beta
        self._eta   = eta

        super(SAE, self).__init__(input_dim, hidden_dim, **kwargs)

    def _collect_summary(self):
        sup_list = super(SAE, self)._collect_summary()
        rl1 = tf.summary.scalar("restoration_loss", self._restoration_loss)
        rl2 = tf.summary.scalar("regularization_loss", self._regularization_loss)
        kl  = tf.summary.scalar("kl_divergence_loss", self._kl_divergence_loss)
        sup_list.extend([rl1, rl2, kl])
        return sup_list

    def _define_loss(self):
        with tf.variable_scope("losses"):
            with tf.variable_scope("restoration_loss"):
                rest_loss = super(SAE, self)._get_restoration_loss()
            with tf.variable_scope("regularization_loss"):
                regu_loss = self._get_regularization_loss()
            with tf.variable_scope("kl_divergence_loss"):
                kl_loss   = self._get_kl_divergence_loss()

            loss = rest_loss + self._alpha * regu_loss + self._beta * kl_loss
            loss = tf.identity(loss, "total_loss")

        self._restoration_loss    = rest_loss
        self._regularization_loss = regu_loss
        self._kl_divergence_loss  = kl_loss
        self._loss = loss

    def _get_regularization_loss(self):
        regu_loss = tf.reduce_sum(self._encoder_weight**2) + tf.reduce_sum(self._decoder_weight**2)
        regu_loss /= 2.0
        regu_loss = tf.identity(regu_loss, "regularization_loss")
        return regu_loss

    def _get_kl_divergence_loss(self):
        hidden_normalized = (1.0 + tf.reduce_mean(self._hidden_layer, axis=0)) / 2.0
        kl_loss = tf.reduce_sum(
                self._eta * tf.log(self._eta) -
                self._eta * tf.log(hidden_normalized) +
                (1.0 - self._eta) * tf.log(1.0 - self._eta) -
                (1.0 - self._eta) * tf.log(1.0 - hidden_normalized),
                name="kl_divergence_loss"
                )
        return kl_loss

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def eta(self):
        return self._eta
