import numpy as np
from .model import SAE, SAE_PBHL

class DSAE(object):

    def __init__(self, structure, alpha=0.003, beta=0.7, eta=0.5):
        alpha = float(alpha)
        beta  = float(beta)
        eta   = float(eta)
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
    def structure(self):
        return self._params["structure"]

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

    def decode(self, h_feature):
        for network in self._networks[::-1]:
            h_feature = network.decode(h_feature)
        return h_feature

    def feature(self, x_in):
        for network in self._networks:
            x_in = network.feature(x_in)
        return x_in

    def fit(self, x_in, epoch=5, epsilon=0.000001):
        for i, network in enumerate(self._networks):
            network.fit(x_in, epoch=epoch, epsilon=epsilon)
            x_in = network.encode(x_in)
            self._params["encode_W_{}".format(i)] = network.encode_weight
            self._params["encode_b_{}".format(i)] = network.encode_bias
            self._params["decode_W_{}".format(i)] = network.decode_weight
            self._params["decode_b_{}".format(i)] = network.decode_bias

    def save_params(self, f):
        params = self._params
        np.savez(f, **params)

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
        for i, network in enumerate(self._networks):
            network.load_params_by_dict({
                "input_dim": network.input_dim,
                "hidden_dim": network.hidden_dim,
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
        instance = cls(params["structure"])
        instance.load_params_by_dict(params)
        return instance

class DSAE_PBHL(DSAE):

    def _stack_networks(self, structure, alpha, beta, eta):
        for i in range(len(structure)-3):
            n_in = structure[i]
            n_hidden = structure[i+1]
            self._networks.append(SAE(n_in, n_hidden, alpha=alpha, beta=beta, eta=eta))
        self._networks.append(SAE(structure[-3], structure[-2][0], alpha=alpha, beta=beta, eta=eta))
        self._networks.append(SAE_PBHL(structure[-2], structure[-1], alpha=alpha, beta=beta, eta=eta))

    def encode(self, x_in, x_pb):
        for network in self._networks[:-1]:
            x_in = network.encode(x_in)
        return network[-1].encode(x_in, x_pb)

    def decode(self, h_in, h_pb):
        h_in = network[-1].decode(h_in, h_pb)
        for network in self._networks[-2::-1]:
            h_in = network.decode(h_in)
        return h_in

    def feature_pb(self, x_in, x_pb):
        for network in self._networks[:-1]:
            x_in = network.feature(x_in)
        return network[-1].feature_pb(x_in, x_pb)

    def fit(self, x_in, x_pb, epoch=5, epsilon=0.000001):
        pbhl_net = self._networks[-1]
        for i, network in enumerate(self._networks):
            if network is pbhl_net:
                network.fit(x_in, x_pb, epoch=epoch, epsilon=epsilon)
                # x_in = network.encode(x_in, x_pb)
            else:
                network.fit(x_in, epoch=epoch, epsilon=epsilon)
                x_in = network.encode(x_in)
            self._params["encode_W_{}".format(i)] = network.encode_weight
            self._params["encode_b_{}".format(i)] = network.encode_bias
            self._params["decode_W_{}".format(i)] = network.decode_weight
            self._params["decode_b_{}".format(i)] = network.decode_bias

    def save_params(self, f):
        params = _unparse_params_dict(self._params)
        np.savez(f, **params)

    def load_params(self, f):
        params = dict(np.load(f))
        params = _parse_params_dict(params)
        super(DSAE_PBHL, self).load_params_by_dict(self, params)


    @classmethod
    def load(cls, source):
        if type(source) is dict:
            params = source
        else:
            params = dict(np.load(source))
        params = _parse_params_dict(params)
        return super(DSAE_PBHL, cls).load(params)

def _parse_params_dict(dict):
    params = dict.copy()
    structure = list(params.pop("structure_head"))
    structure.append(list(params.pop("structure_input")))
    structure.append(list(params.pop("structure_hidden")))
    params["structure"] = structure
    return params

def _unparse_params_dict(dict):
    params = dict.copy()
    structure = params.pop("structure")
    params["structure_head"] = structure[:-2]
    params["structure_input"] = structure[-2]
    params["structure_hidden"] = structure[-1]
    return params
