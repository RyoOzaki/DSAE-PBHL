import numpy as np
from .model import SAE, SAE_PBHL

class DSAE(object):

    def __init__(self, structure, alpha=0.003, beta=0.7, eta=0.5):
        self._params = {"structure": structure,
            "alpha": alpha, "beta": beta, "eta": eta
            }
        for i in range(len(structure)-1):
            self._params["encode_W_{}".format(i)] = None
            self._params["encode_b_{}".format(i)] = None
            self._params["decode_W_{}".format(i)] = None
            self._params["decode_b_{}".format(i)] = None
        self._networks = []
        self._stack_networks(structure, alpha, beta, eta)

    def _stack_networks(self, structure, alpha, beta, eta):
        for i in range(len(structure)-1):
            n_in = structure[i]
            n_hidden = structure[i+1]
            self._networks.append(SAE(n_in, n_hidden, alpha=alpha, beta=beta, eta=eta))

    @property
    def encode_weights(self):
        return [self._params["encode_W_{}".format(i)] for i in range(len(structure)-1)]

    @property
    def encode_biases(self):
        return [self._params["encode_b_{}".format(i)] for i in range(len(structure)-1)]

    @property
    def decode_weights(self):
        return [self._params["decode_W_{}".format(i)] for i in range(len(structure)-1)]

    @property
    def decode_biases(self):
        return [self._params["decode_b_{}".format(i)] for i in range(len(structure)-1)]

    @property
    def networks(self):
        return self._networks

    def encode(self, x_in):
        for network in self._networks:
            x_in = network.encode(x_in)
        return x_in

    def decode(self, h_in):
        for network in self._networks[::-1]:
            h_in = network.decode(h_in)
        return h_in

    def feature(self, x_in):
        for network in self._networks[:-1]:
            x_in = network.encode(x_in)
        return self._networks[-1].feature(x_in)

    def fit(self, x_train, epoch=5, epsilon=0.000001):
        for i, network in enumerate(self._networks):
            network.fit(x_train, epoch=epoch, epsilon=epsilon)
            self._params["encode_W_{}".format(i)] = network.encode_weight
            self._params["encode_b_{}".format(i)] = network.encode_bias
            self._params["decode_W_{}".format(i)] = network.decode_weight
            self._params["decode_b_{}".format(i)] = network.decode_bias
            x_train = network.encode(x_train)

    def save_params(self, f):
        np.savez(f, **self._params)

    def load_params(self, f):
        params = np.load(f)
        self.load_params_by_dict(self, params)

    def load_params_by_dict(self, dic):
        assert "structure" in dic
        # if "structure" not in dic:
        #     raise RuntimeError("Does not have 'structure'.")
        self._params = dic
        structure = self._params["structure"]
        self._networks = []
        self._stack_networks(structure, self._params["alpha"], self._params["beta"], self._params["eta"])
        for network in self._networks[:-1]:
            network.load_params_by_dict({
                "input_dim": n_in,
                "hidden_dim": n_hidden,
                "alpha": self._params["alpha"],
                "beta": self._params["beta"],
                "eta": self._params["eta"],
                "encode_W": self._params["encode_W_{}".format(i)],
                "encode_b": self._params["encode_b_{}".format(i)],
                "decode_W": self._params["decode_W_{}".format(i)],
                "decode_b": self._params["decode_b_{}".format(i)]
            })

    @classmethod
    def load(cls, source):
        if type(source) is dict:
            params = source
        else:
            params = np.load(source)
        assert "structure" in params
        # if "structure" not in params:
        #     raise RuntimeError("Does not have 'structure'.")
        instance = cls(structure)
        instance.load_params_by_dict(f)
        return instance

class DSAE_PBHL(DSAE):

    def _stack_networks(self, structure, alpha, beta, eta):
        for i in range(len(structure)-3):
            n_in = structure[i]
            n_hidden = structure[i+1]
            self._networks.append(SAE(n_in, n_hidden, alpha=alpha, beta=beta, eta=eta))
        self._networks.append(SAE(structure[-3], structure[-2][0], alpha=alpha, beta=beta, eta=eta))
        self._networks.append(SAE_PBHL(strucutre[-2], structure[-1], alpha=alpha, beta=beta, eta=eta))

    def fit(self, x_train, x_pb, epoch=5, epsilon=0.000001):
        pbhl_net = self._networks[-1]
        for i, network in enumerate(self._networks):
            if network is pbhl_net:
                network.fit(x_train, x_pb, epoch=epoch, epsilon=epsilon)
            else:
                network.fit(x_train, epoch=epoch, epsilon=epsilon)
            self._params["encode_W_{}".format(i)] = network.encode_weight
            self._params["encode_b_{}".format(i)] = network.encode_bias
            self._params["decode_W_{}".format(i)] = network.decode_weight
            self._params["decode_b_{}".format(i)] = network.decode_bias
            x_train = network.encode(x_train)
