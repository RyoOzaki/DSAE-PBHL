import tensorflow as tf

def weight_variable(shape, **kwargs):
    # initial = tf.random_normal(shape, stddev=0.1)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, **kwargs)

def bias_variable(shape, **kwargs):
    # initial = tf.random_normal(shape, stddev=0.1)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, **kwargs)
