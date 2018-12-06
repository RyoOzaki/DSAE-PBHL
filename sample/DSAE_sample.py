import numpy as np
import tensorflow as tf
from pathlib import Path

from DSAE_PBHL import DSAE
from DSAE_PBHL.util import Normalizer

import matplotlib.pyplot as plt

# -------------------DATA loading...
def packing(np_objs):
    return np.concatenate(np_objs, axis=0)

def unpacking(np_obj, lengths):
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    N = len(lengths)
    return [np_obj[cumsum_lens[i]:cumsum_lens[i+1]] for i in range(N)]

data_npz     = np.load("DATA/data.npz")
data_names   = data_npz.files
data_aranged = [data_npz[name] for name in data_names]
data_lengths = np.array([data.shape[0] for data in data_aranged])

data_unnormalized = packing(data_aranged)
normalizer = Normalizer()
data = normalizer.normalize(data_unnormalized)

with tf.variable_scope("DSAE_sample"):
    dsae = DSAE([12, 8, 5, 3])

T = 100
epoch = 10
global_step = 0
L = len(dsae.networks)
ckpt_file = "ckpt/model.ckpt"
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
with tf.Session() as sess:
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    for idx in range(L):
        summary_writer = tf.summary.FileWriter(f"graph/{idx+1}_th_network", sess.graph)
        print(f"Training {idx+1} th network.")
        for t in range(T):
            # print(f"{global_step}: {idx+1}/{L}, {t+1}/{T}")
            saver.save(sess, ckpt_file, global_step=global_step)
            for _ in range(epoch):
                dsae.fit(sess, idx, data, epoch, summary_writer=summary_writer)
            global_step += epoch
    feature = dsae.hidden_layers_with_eval(sess, data)[-1]

plt.subplot(1, 2, 1)
plt.plot(data[:100])
plt.xlabel("Input features")

plt.subplot(1, 2, 2)
plt.plot(feature[:100])
plt.xlabel("Compressed features")

plt.show()
