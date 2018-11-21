import numpy as np
import tensorflow as tf
from pathlib import Path

from DSAE_PBHL import DSAE, DSAE_PBHL
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
data_pb = np.random.randint(2, size=(data.shape[0], 2))

load_parameters = False
graph = tf.Graph()
with graph.as_default():
    network = DSAE_PBHL([12, 8, 5, 3], [2, 1], pb_activator=tf.nn.softmax)
    network.init_session()
    if load_parameters:
        network.load_variables()
    else:
        network.initialize_variables()
        network.fit(data, data_pb, epoch=10, epsilon=0.01)
    encode = network.encode(data, data_pb)
    feature = network.feature(data)
    decode = network.decode(encode[:, :3], encode[:, 3:])
    network.close_session()

plt.subplot(1, 2, 1)
plt.plot(data[:100])
plt.xlabel("Input features")

plt.subplot(1, 2, 2)
plt.plot(feature[:100])
plt.xlabel("Compressed features")

plt.show()

plt.clf()

plt.subplot(1, 2, 1)
plt.plot(data[:100])
plt.xlabel("Input features")

plt.subplot(1, 2, 2)
plt.plot(decode[:100])
plt.xlabel("Restorated features")

plt.show()
