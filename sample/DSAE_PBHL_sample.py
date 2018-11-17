import numpy as np
import tensorflow as tf
from pathlib import Path

from DSAE_PBHL import DSAE, DSAE_PBHL
from DSAE_PBHL.util import Normalizer

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
# network = DSAE([12, 8, 5, 3])
network = DSAE_PBHL([12, 8, 5, 3], [2, 1], pb_activator=tf.nn.softmax)
network.init_session()
if load_parameters:
    network.load_variables()
else:
    network.initialize_variables()
network.fit(data, data_pb, epoch=10, epsilon=0.01)
feature = network.feature(data)
network.close_session()

tf.reset_default_graph()
# If you want to define tf.session expressly
network = DSAE_PBHL([12, 8, 5, 3], [2, 1], pb_activator=tf.nn.softmax)
load_parameters = True
with tf.Session() as sess:
    network.set_session(sess)
    if load_parameters:
        network.load_variables()
    else:
        network.initialize_variables()
    network.fit(data, data_pb, epsilon=0.001, summary_prefix="graph2")
    feature = network.feature(data)
network.close_session()
