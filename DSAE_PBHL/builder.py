from .Model import Model, PB_Model
from .Deep_Model import Deep_Model, Deep_PB_Model

class Builder(object):

    def __init__(self, input_dim, pb_input_dim=-1):
        self._network_class = []
        self._network_node = [input_dim, ]
        self._network_kwargs = []
        self._pb_input_dim = pb_input_dim
        self._pb_hidden_dim = -1
        self._has_pb_net = False

    def stack(self, network_class, hidden_dim, pb_hidden_dim=-1, **kwargs):
        assert issubclass(network_class, (Model, PB_Model))
        if issubclass(network_class, PB_Model):
            assert not self._has_pb_net
            assert pb_hidden_dim > 0
            self._has_pb_net = True
            self._pb_hidden_dim = pb_hidden_dim
        self._network_class.append(network_class)
        self._network_node.append(hidden_dim)
        self._network_kwargs.append(kwargs)

    def build(self):
        if self._pb_input_dim > 0:
            assert self._has_pb_net
            obj = Deep_PB_Model(
                self._network_node,
                [self._pb_input_dim, self._pb_hidden_dim],
                self._network_class,
                self._network_kwargs
                )
        else:
            obj = Deep_Model(
                self._network_node,
                self._network_class,
                self._network_kwargs
                )
        return obj

    # Default format: "{tab}{class_name}[{input_dim} -> {hidden_dim} -> {input_dim}, {kwargs}]"
    # index, tab, class_name, kwargs, input_dim, hidden_dim
    def print_recipe(self, format=None, tab_str="  "):
        if format is None:
            format = "{tab}{class_name}[{input_dim} -> {hidden_dim} -> {input_dim}, {kwargs}]"
        for i, class_obj in enumerate(self._network_class):
            class_name = class_obj.__name__
            kwargs = self._network_kwargs[i]
            if issubclass(class_obj, PB_Model):
                input_feature_dim = self._network_node[i]
                input_pb_dim = self._pb_input_dim
                hidden_feature_dim = self._network_node[i+1]
                hidden_pb_dim = self._pb_hidden_dim
                input_dim = f"({input_feature_dim} + {input_pb_dim})"
                hidden_dim = f"({hidden_feature_dim} + {hidden_pb_dim})"
            else:
                input_dim = self._network_node[i]
                hidden_dim = self._network_node[i+1]
            tab = tab_str * i

            print(format.format(tab=tab, index=i, class_name=class_name, kwargs=kwargs, input_dim=input_dim, hidden_dim=hidden_dim))
