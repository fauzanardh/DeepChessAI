import json
import numpy as np


def create_uci_labels():
    labels = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    nums = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    # UCI labels magic xd
    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                           [(l1, t) for t in range(8)] + \
                           [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                           [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + nums[n1] + letters[l2] + nums[n2]
                    labels.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels.append(l + '2' + l + '1' + p)
            labels.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_s1 = letters[l1 - 1]
                labels.append(l + '2' + l_s1 + '1' + p)
                labels.append(l + '7' + l_s1 + '8' + p)
            if l1 < 7:
                l_p1 = letters[l1 + 1]
                labels.append(l + '2' + l_p1 + '1' + p)
                labels.append(l + '7' + l_p1 + '8' + p)
    return labels


def flipped_uci_labels():
    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])
    return [repl(x) for x in create_uci_labels()]


class ModelConfig(object):
    def __init__(self):
        # These are the default value
        self.cnn_filter_num = 256
        self.cnn_first_filter_size = 5
        self.cnn_filter_size = 3
        self.input_depth = 18
        self.l2_reg = 1e-4
        self.res_layer_num = 7
        self.value_fc_size = 256


class Config(object):
    labels = create_uci_labels()
    n_labels = len(labels)
    flipped_labels = flipped_uci_labels()
    unflipped_labels = None

    def __init__(self, path: str = None):
        if path is not None:
            with open(path, 'r') as fp:
                _json_config = json.load(fp)
            self.model_path = _json_config["model_path"]
            self.model = ModelConfig()
            self.model.cnn_filter_num = _json_config["model"]["cnn_filter_num"]
            self.model.cnn_first_filter_size = _json_config["model"]["cnn_first_filter_size"]
            self.model.cnn_filter_size = _json_config["model"]["cnn_filter_size"]
            self.model.input_depth = _json_config["model"]["input_depth"]
            self.model.l2_reg = _json_config["model"]["l2_reg"]
            self.model.res_layer_num = _json_config["model"]["res_layer_num"]
            self.model.value_fc_size = _json_config["model"]["value_fc_size"]
        else:
            self.model_path = ""
            self.model = ModelConfig()

    @staticmethod
    def flip_policy(policy):
        return np.asarray([policy[i] for i in Config.unflipped_index])

    def save_config(self, path: str):
        with open(path, "w+") as fp:
            json.dump(self, fp, default=lambda o: o.__dict__, sort_keys=True, indent=4)


Config.unflipped_index = [Config.labels.index(x) for x in Config.flipped_labels]
