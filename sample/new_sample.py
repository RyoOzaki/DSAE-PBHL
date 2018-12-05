import tensorflow as tf
import numpy as np

from DSAE_PBHL import Builder
from DSAE_PBHL import AE, SAE, SAE_PBHL
from DSAE_PBHL import DAE, DSAE, DSAE_PBHL
from DSAE_PBHL import DSAE_Soft, DSAE_PBHL_Soft

N = 100
dim_f = 30
dim_p = 3
input = np.random.rand(N, dim_f) * 2 - 1
input_pb = np.identity(dim_p)[np.random.randint(dim_p, size=N)]

with tf.variable_scope("Auto-encoder_sample"):
    # dim_f is 30.
    DAE_1 = DAE([30, 10, 5])
    # Define a deep auto-encoder.
    # The structure is as follows.
    # input(30) => AE(30 -> 10 -> 30)
    #                        |-hidden(10) => AE(10 -> 5 -> 10)
    #                                                 |-hidden(5)

# If you need to define another network in 1 code, please use tf.variable_scope
#   and set another scope name.
# Because, the variable name may collision.
with tf.variable_scope("Auto-encoder_sample2"):
    DAE_2 = DAE([30, 10, 5])

with tf.Session() as sess:
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    # Training network which structure is "30 -> 10 -> 30."
    # The return values are (epoch of trained network, loss of trained network, tf summary object)
    res_1 = DAE_1.fit(sess, 0, input, 100)
    res_2 = DAE_2.fit(sess, 0, input, 100)

    print(res_1[:-1])
    print(res_2[:-1])

tf.reset_default_graph()
#===============================================================================

# If you want to make a more complex model, please use builder in DSAE_PBHL.util.
# The builder.stack() saves a recipe of complex network, and builder.build() makes a network by the recipe.
# Therefore, if you need to use the tf.variable_scope, you need to apply it to the builder.build().
# In addition, the builder is saving the recipe after called builder.build(), so, you can make the same structure network using builder.build().
# But, you can not call builder.build() over 2 times in the same tf.variable_scope.

builder = Builder(30, pb_input_dim=3)
builder.stack(SAE, 10, beta=0.1)
builder.stack(SAE_PBHL, 5, pb_hidden_dim=1, alpha=10)
builder.stack(SAE, 2, beta=1, alpha=0.09)
with tf.variable_scope("Builder_sample_1"):
    builder_dsae_1 = builder.build()
with tf.variable_scope("Builder_sample_2"):
    builder_dsae_2 = builder.build()

builder.print_recipe()

tf.reset_default_graph()
#===============================================================================

# If you use the tensorboard
builder = Builder(30, pb_input_dim=3)
builder.stack(SAE, 10, beta=0.1)
builder.stack(SAE_PBHL, 5, pb_hidden_dim=1, alpha=10)
builder.stack(SAE, 2, beta=1, alpha=0.09)
with tf.variable_scope("DSAE_PBHL_sample"):
    dsae_pbhl = builder.build()

with tf.Session() as sess:
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    N = len(dsae_pbhl.networks)
    for idx in range(N):
        summary_writer = tf.summary.FileWriter(f"graph/{idx+1}_th_network", sess.graph)
        for _ in range(100):
            dsae_pbhl.fit(sess, idx, input, input_pb, 10, summary_writer=summary_writer)
            # or
            # step, loss, summary = dsae_pbhl.fit(sess, idx, input, input_pb, 10)
            # summary_writer.add_summary(summary, step)
        summary_writer.close()

tf.reset_default_graph()
#===============================================================================

# If you use the saver
with tf.variable_scope("DSAE_PBHL_sample"):
    dsae = DSAE_PBHL([30, 10, 5], [3, 2])

ckpt_file = "ckpt/model.ckpt"
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
global_step = 0
epoch = 10
with tf.Session() as sess:
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    N = len(dsae.networks)
    for idx in range(N):
        for _ in range(100):
            saver.save(sess, ckpt_file, global_step=global_step)
            dsae.fit(sess, idx, input, input_pb, epoch)
            global_step += epoch

tf.reset_default_graph()
