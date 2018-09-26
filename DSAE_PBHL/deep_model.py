from model import SAE, SAE_PBHL

class DSAE(object):

    def __init__(self, structure, alpha=0.003, beta=0.7, eta=0.5):
        self.params = {"structure": structure,
            "alpha": alpha, "beta": beta, "eta": eta
            }
        for i in range(len(structure)-1):
            self.params["encode_W_{}".format(i)] = None
            self.params["encode_b_{}".format(i)] = None
            self.params["decode_W_{}".format(i)] = None
            self.params["decode_b_{}".format(i)] = None
        self.networks = []
        self._stack_networks(self, structure, alpha, beta, eta)

    def _stack_networks(self, structure, alpha, beta, eta):
        for i in range(len(structure)-1):
            n_in = structure[i]
            n_hidden = structure[i+1]
            self.networks.append(SAE(n_in, n_hidden, alpha=alpha, beta=beta, eta=eta))

    @property
    def encode_weights(self):
        return [self.params["encode_W_{}".format(i)] for i in range(len(structure)-1)]

    @property
    def encode_biases(self):
        return [self.params["encode_b_{}".format(i)] for i in range(len(structure)-1)]

    @property
    def decode_weights(self):
        return [self.params["decode_W_{}".format(i)] for i in range(len(structure)-1)]

    @property
    def decode_biases(self):
        return [self.params["decode_b_{}".format(i)] for i in range(len(structure)-1)]


    def encode(self, x_in):
        for network in self.networks:
            x_in = network.encode(x_in)
        return x_in

    def decode(self, h_in):
        for network in self.networks[::-1]:
            h_in = network.decode(h_in)
        return h_in

    def feature(self, x_in):
        for network in self.networks[:-1]:
            x_in = network.encode(x_in)
        return self.networks[-1].feature(x_in)

    def fit(self, x_train, epoch=5, epsilon=0.000001):
        for i, network in enumerate(self.networks):
            network.fit(x_train, epoch=epoch, epsilon=epsilon)
            self.params["encode_W_{}".format(i)] = network.encode_weight
            self.params["encode_b_{}".format(i)] = network.encode_bias
            self.params["decode_W_{}".format(i)] = network.decode_weight
            self.params["decode_b_{}".format(i)] = network.decode_bias
            x_train = network.encode(x_train)

    def save_params(self, f):
        np.savez(f, **self.params)

    def load_params(self, f):
        params = np.load(f)
        self.load_params_by_dict(self, params)

    def load_params_by_dict(self, dic):
        if "structure" not in dic:
            raise RuntimeError("Does not have 'structure'.")
        self.params = dic
        structure = self.params["structure"]
        self.networks = []
        self._stack_networks(structure, self.params["alpha"], self.params["beta"], self.params["eta"])
        for network in self.networks[:-1]:
            network.load_params_by_dict({
                "input_dim": n_in,
                "hidden_dim": n_hidden,
                "alpha": self.params["alpha"],
                "beta": self.params["beta"],
                "eta": self.params["eta"],
                "encode_W": self.params["encode_W_{}".format(i)],
                "encode_b": self.params["encode_b_{}".format(i)],
                "decode_W": self.params["decode_W_{}".format(i)],
                "decode_b": self.params["decode_b_{}".format(i)]
            })

    @classmethod
    def load(cls, source):
        if type(source) is dict:
            params = source
        else:
            params = np.load(source)
        if "structure" not in params:
            raise RuntimeError("Does not have 'structure'.")
        instance = cls(structure)
        instance.load_params_by_dict(f)
        return instance

class DSAE_PBHL(DSAE):
    
    def _stack_networks(self, structure, alpha, beta, eta):
        for i in range(len(structure)-2):
            n_in = structure[i]
            n_hidden = structure[i+1]
            self.networks.append(SAE(n_in, n_hidden, alpha=alpha, beta=beta, eta=eta))
        self.networks.append(SAE(n_in, n_hidden, alpha=alpha, beta=beta, eta=eta))
