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


class PlayConfig(object):
    def __init__(self):
        self.c_puct = 1.5
        self.dirichlet_alpha = 0.3
        self.max_game_length = 1000
        self.max_game_per_file = 32
        self.max_processes = 1
        self.min_resign_turn = 5
        self.noise_eps = 0.25
        self.resign_threshold = -0.8
        self.search_threads = 16
        self.simulation_num_per_move = 100
        self.tau_decay_rate = 0.99
        self.virtual_loss = 3


class SupervisedLearning(object):
    def __init__(self):
        self.max_processes = 4
        self.min_elo_policy = 1200
        self.max_elo_policy = 2800


class TrainingConfig(object):
    def __init__(self):
        self.max_processes = 1
        self.batch_size = 128
        self.epoch_size = 0
        self.epoch_to_checkpoint = 1
        self.dataset_size = 100000
        self.start_total_steps = 0
        self.save_model_steps = 25
        self.load_data_steps = 100
        self.loss_weight = [1.25, 1.0]


class EvaluateConfig(object):
    def __init__(self):
        self.c_puct = 1
        self.game_num = 50
        self.max_processes = 1
        self.noise_eps = 0
        self.replace_rate = 0.55
        self.search_threads = 16
        self.simulation_num_per_move = 200
        self.tau_decay_rate = 0.6


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
            self.play_path = _json_config["play_path"]
            self.pgn_path = _json_config["pgn_path"]
            self.tfr_path = _json_config["tfr_path"]

            # Config for the Chess AI Model
            self.model = ModelConfig()
            self.model.cnn_filter_num = _json_config["model"]["cnn_filter_num"]
            self.model.cnn_first_filter_size = _json_config["model"]["cnn_first_filter_size"]
            self.model.cnn_filter_size = _json_config["model"]["cnn_filter_size"]
            self.model.input_depth = _json_config["model"]["input_depth"]
            self.model.l2_reg = _json_config["model"]["l2_reg"]
            self.model.res_layer_num = _json_config["model"]["res_layer_num"]
            self.model.value_fc_size = _json_config["model"]["value_fc_size"]

            # Config for the Player
            self.play = PlayConfig()
            self.play.c_puct = _json_config["play"]["c_puct"]
            self.play.dirichlet_alpha = _json_config["play"]["dirichlet_alpha"]
            self.play.max_game_length = _json_config["play"]["max_game_length"]
            self.play.max_game_per_file = _json_config["play"]["max_game_per_file"]
            self.play.max_processes = _json_config["play"]["max_processes"]
            self.play.min_resign_turn = _json_config["play"]["min_resign_turn"]
            self.play.noise_eps = _json_config["play"]["noise_eps"]
            self.play.resign_threshold = _json_config["play"]["resign_threshold"]
            self.play.search_threads = _json_config["play"]["search_threads"]
            self.play.simulation_num_per_move = _json_config["play"]["simulation_num_per_move"]
            self.play.tau_decay_rate = _json_config["play"]["tau_decay_rate"]
            self.play.virtual_loss = _json_config["play"]["virtual_loss"]

            # Config for supervised learning
            self.supervised_learning = SupervisedLearning()
            self.supervised_learning.max_processes = _json_config["supervised_learning"]["max_processes"]
            self.supervised_learning.min_elo_policy = _json_config["supervised_learning"]["min_elo_policy"]
            self.supervised_learning.max_elo_policy = _json_config["supervised_learning"]["max_elo_policy"]

            # Config for training
            self.training = TrainingConfig()
            self.training.max_processes = _json_config["training"]["max_processes"]
            self.training.batch_size = _json_config["training"]["batch_size"]
            self.training.epoch_size = _json_config["training"]["epoch_size"]
            self.training.epoch_to_checkpoint = _json_config["training"]["epoch_to_checkpoint"]
            self.training.dataset_size = _json_config["training"]["dataset_size"]
            self.training.start_total_steps = _json_config["training"]["start_total_steps"]
            self.training.save_model_steps = _json_config["training"]["save_model_steps"]
            self.training.load_data_steps = _json_config["training"]["load_data_steps"]
            self.training.loss_weight = _json_config["training"]["loss_weight"]

            # Config for evaluator
            self.evaluate = EvaluateConfig()
            self.evaluate.c_puct = _json_config["evaluate"]["c_puct"]
            self.evaluate.game_num = _json_config["evaluate"]["game_num"]
            self.evaluate.max_processes = _json_config["evaluate"]["max_processes"]
            self.evaluate.noise_eps = _json_config["evaluate"]["noise_eps"]
            self.evaluate.replace_rate = _json_config["evaluate"]["replace_rate"]
            self.evaluate.search_threads = _json_config["evaluate"]["search_threads"]
            self.evaluate.simulation_num_per_move = _json_config["evaluate"]["simulation_num_per_move"]
            self.evaluate.tau_decay_rate = _json_config["evaluate"]["tau_decay_rate"]
        else:
            self.model_path = ""
            self.play_path = ""
            self.pgn_path = ""
            self.tfr_path = ""
            self.model = ModelConfig()
            self.play = PlayConfig()
            self.supervised_learning = SupervisedLearning()
            self.training = TrainingConfig()
            self.evaluate = EvaluateConfig()

    @staticmethod
    def flip_policy(policy):
        return np.asarray([policy[i] for i in Config.unflipped_index])

    def save_config(self, path: str):
        with open(path, "w+") as fp:
            fp.write(json.dumps(self, default=lambda o: o.__dict__, indent=4))


Config.unflipped_index = [Config.labels.index(x) for x in Config.flipped_labels]
