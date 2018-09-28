import numpy as np
import tensorflow as tf
from pathlib import Path

from DSAE_PBHL.model import SAE, SAE_PBHL
from DSAE_PBHL.deep_model import DSAE, DSAE_PBHL
from DSAE_PBHL.util.normalizer import Normalizer

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

# -------------------
structure = [12, 8, 5, 3]

dsae = DSAE(structure)
dsae.fit(data)
encoded_packed = dsae.encode(data)
encoded_unpacked = unpacking(encoded_packed, data_lengths)
encoded_dic = {name: encoded_data for name, encoded_data in zip(data_names, encoded_unpacked)}
np.savez("DSAE_encoded.npz")
dsae.save_params("DSAE_params.npz")


-------------------
structure = [12, 8, [5, 2], [3, 1]]
data_pb = np.zeros((data.shape[0], 2))

dsae_pbhl = DSAE_PBHL(structure)
dsae_pbhl.fit(data, data_pb)
feature_packed = dsae_pbhl.feature(data)
feature_unpacked = unpacking(feature_packed, data_lengths)
feature_dic = {name: feature for name, feature in zip(data_names, feature_unpacked)}
np.savez("DSAE_PBHL_feature.npz")
dsae_pbhl.save_params("DSAE_PBHL_params.npz")
