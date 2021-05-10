from pathlib import Path

import tensorflow as tf
import numpy as np

import chess.pgn

from agent.player import ChessPlayer
from config import Config
from env.chess_env import ChessEnv, Winner, canon_input_planes, is_white_turn, evaluate
from util.data_helper import get_pgn_filenames


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class TFRecordExporter(object):
    def __init__(self, dataset_name, config: Config):
        self.config = config
        self.dataset_name = dataset_name
        self.tfr_dir = Path(self.config.tfr_path)
        tfr_opt = tf.io.TFRecordOptions(
            compression_type="GZIP"
        )
        tfr_file = self.tfr_dir / f"{dataset_name}.tfrecords"
        self.tfr_writer = tf.io.TFRecordWriter(str(tfr_file), tfr_opt)
        self.game_idx = 0

    def close(self):
        self.tfr_writer.close()

    def add_data(self, game):
        self.game_idx += 1
        env, data = get_buffer(self.config, game)
        print(
            f"game {self.game_idx:05} "
            f"halfmoves={env.num_halfmoves:03} {env.winner:12} "
            f"{'by resign' if env.is_resigned else ''} "
            f"| {env.observation}"
        )
        _data = convert_data(data)
        state = _data[0].reshape(-1)
        policy = _data[1].reshape(-1)
        value = _data[2].reshape(-1)
        ex = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "state": _float_feature(state),
                    "policy": _float_feature(policy),
                    "value": _float_feature(value)
                }
            )
        )
        self.tfr_writer.write(ex.SerializeToString())


def get_games_from_pgn(filename):
    pgn = open(filename, "r", errors='ignore')
    offsets = []
    while True:
        offset = pgn.tell()
        headers = chess.pgn.read_headers(pgn)
        if headers is None:
            break
        offsets.append(offset)
    offsets = offsets
    n = len(offsets)
    print(f"Found {n} games!")
    games = []
    for offset in offsets:
        pgn.seek(offset)
        games.append(chess.pgn.read_game(pgn))
    return games


def clip_elo_policy(config, elo):
    return min(1, max(0, elo - config.supervised_learning.min_elo_policy) /
               (config.supervised_learning.max_elo_policy - config.supervised_learning.min_elo_policy))


def get_buffer(config: Config, game):
    env = ChessEnv().reset()
    white = ChessPlayer(config, dummy=True)
    black = ChessPlayer(config, dummy=True)
    result = game.headers["Result"]
    if "WhiteElo" not in game.headers:
        game.headers["WhiteElo"] = "2500"
    if "BlackElo" not in game.headers:
        game.headers["BlackElo"] = "2500"
    white_elo, black_elo = int(game.headers["WhiteElo"]), int(game.headers["BlackElo"])
    white_weight = clip_elo_policy(config, white_elo)
    black_weight = clip_elo_policy(config, black_elo)

    actions = []
    while not game.is_end():
        game = game.variation(0)
        actions.append(game.move.uci())

    k = 0
    while not env.done and k < len(actions):
        if env.white_to_move:
            action = white.sl_action(env.observation, actions[k], weight=white_weight)
        else:
            action = black.sl_action(env.observation, actions[k], weight=black_weight)
        env.step(action, False)
        k += 1

    if not env.board.is_game_over() and result != '1/2-1/2':
        env.is_resigned = True

    if result == "1-0":
        env.winner = Winner.white
        white_win = 1
    elif result == "0-1":
        env.winner = Winner.black
        white_win = -1
    else:
        env.winner = Winner.draw
        white_win = 0

    white.finish_game(white_win)
    black.finish_game(-white_win)

    data = []
    for i in range(len(white.moves)):
        data.append(white.moves[i])
        if i < len(black.moves):
            data.append(black.moves[i])
    return env, data


def convert_data(data):
    state_list = []
    policy_list = []
    value_list = []
    for state_fen, policy, value in data:
        state_planes = canon_input_planes(state_fen)

        if not is_white_turn(state_fen):
            policy = Config.flip_policy(policy)

        move_number = int(state_fen.split(' ')[5])
        value_certainty = min(5, move_number) / 5
        sl_value = value * value_certainty + evaluate(state_fen, False) * (1 - value_certainty)

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(sl_value)

    return np.asarray(state_list, dtype=np.float32), \
           np.asarray(policy_list, dtype=np.float32), \
           np.asarray(value_list, dtype=np.float32)


if __name__ == "__main__":
    _config = Config("config-default.json")
    files = get_pgn_filenames(_config)
    try:
        for file in files:
            _dataset_name = str(file.name).split('.')[0]
            exporter = TFRecordExporter(_dataset_name, _config)
            _games = get_games_from_pgn(file)
            for _game in _games:
                exporter.add_data(_game)
    except KeyboardInterrupt:
        print("Exiting...")
    print(f"Added {exporter.game_idx} games!")
    exporter.close()

