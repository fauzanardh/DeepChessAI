import os
import re
from datetime import datetime
from pathlib import Path
from threading import Thread

import chess.pgn

from agent.player import ChessPlayer
from config import Config
from env.chess_env import ChessEnv, Winner
from util.data_helper import get_pgn_filenames, write_data

TAG_REGEX = re.compile(r"^\[([A-Za-z0-9_]+)\s+\"(.*)\"\]\s*$")


def start(config: Config):
    return SupervisedLearning(config).start()


class SupervisedLearning(object):
    def __init__(self, config: Config):
        self.config = config
        self.buffer = []
        self.games = []
        self.game_idx = 0

    def start(self):
        self.get_pgn_files()
        for game in self.games:
            self.game_idx += 1
            env, wm, bm = get_buffer(self.config, game)
            data = []
            for i in range(len(wm)):
                data.append(wm[i])
                if i < len(bm):
                    data.append(bm[i])
            del wm, bm
            self.buffer.extend(data)
            del data
            print(
                f"game {self.game_idx:05} "
                f"halfmoves={env.num_halfmoves:03} {env.winner:12} "
                f"{'by resign' if env.is_resigned else ''} "
                f"| {env.observation}"
            )
            if self.game_idx % self.config.play.max_game_per_file == 0:
                self.flush_buffer()
        # with ProcessPoolExecutor(max_workers=self.config.supervised_learning.max_processes) as executor:
        #     futures = [executor.submit(get_buffer, self.config, game) for game in games]
        #     for future in as_completed(futures):
        #         self.game_idx += 1
        #         env, wm, bm = future.result()
        #         data = []
        #         for i in range(len(wm)):
        #             data.append(wm[i])
        #             if i < len(bm):
        #                 data.append(bm[i])
        #         del wm, bm
        #         self.buffer.extend(data)
        #         del data
        #         print(
        #             f"game {self.game_idx:05} "
        #             f"halfmoves={env.num_halfmoves:03} {env.winner:12} "
        #             f"{'by resign' if env.is_resigned else ''} "
        #             f"| {env.observation}"
        #         )
        #         if self.game_idx % self.config.play.max_game_per_file == 0:
        #             self.flush_buffer()
        if len(self.buffer) > 0:
            self.flush_buffer()

    def get_pgn_files(self):
        files = get_pgn_filenames(self.config)
        for file in files:
            self.games.extend(get_games_from_pgn(file))
            print(f"Current total games: {len(self.games)}")


    def flush_buffer(self):
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(self.config.play_path, f"play_{game_id}.json")
        Path(self.config.play_path).mkdir(exist_ok=True)
        buffer = self.buffer.copy()
        self.buffer = []
        write_data(path, buffer)
        # thread = Thread(target=write_data, args=(path, buffer))
        # thread.start()


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

    # data = []
    # for i in range(len(white.moves)):
    #     data.append(white.moves[i])
    #     if i < len(black.moves):
    #         data.append(black.moves[i])
    return env, white.moves, black.moves


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
