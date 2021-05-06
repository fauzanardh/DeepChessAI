import json
import os
from datetime import datetime
from time import time

from agent.model import ChessModel
from agent.player import ChessPlayer
from config import Config
from env.chess_env import ChessEnv, Winner


def start(config: Config):
    return SelfPlay(config).start()


class SelfPlay(object):
    def __init__(self, config: Config):
        self.config = config
        self.agent = self.load_model()
        self.buffer = []

    def load_model(self):
        agent = ChessModel(self.config)
        agent.load_latest()
        agent.model.summary()
        return agent

    def start(self):
        self.buffer = []
        # futures = deque()
        # with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
        #     for game_idx in range(self.config.play.max_processes * 2):
        #         futures.append(executor.submit(self_play_buffer, self.config, self.agent))
        #     game_idx = 0
        #     while game_idx < self.config.play.max_total_game:
        #         game_idx += 1
        #         start_time = time()
        #         env, data = futures.popleft().result()
        #         print(
        #             f"game {game_idx:05} time={time() - start_time:5.1f}s "
        #             f"halfmoves={env.num_halfmoves:03} {env.winner:12} "
        #             f"{'by resign' if env.resigned else ''}"
        #         )
        #         self.buffer += data

        game_idx = 0
        while game_idx < self.config.play.max_total_game:
            game_idx += 1
            start_time = time()
            env, data = self_play_buffer(self.config, self.agent)
            print(env.board.fen())
            print(
                f"game {game_idx:05} time={time() - start_time:5.1f}s "
                f"halfmoves={env.num_halfmoves:03} {env.winner:12} "
                f"{'by resign' if env.is_resigned else ''}"
            )
            self.buffer += data

        if len(self.buffer) > 0:
            self.flush_buffer()

    def flush_buffer(self):
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(self.config.play_path, f"play_{game_id}.json")
        with open(path, "w+") as fp:
            json.dump(self.buffer, fp)
        self.buffer = []


def self_play_buffer(config: Config, agent: ChessModel):
    env = ChessEnv().reset()

    white = ChessPlayer(config, agent)
    black = ChessPlayer(config, agent)

    while not env.done:
        if env.white_to_move:
            action = white.action(env)
        else:
            action = black.action(env)
        env.step(action)
        if env.num_halfmoves >= config.play.max_game_length:
            env.adjudicate()

    if env.winner == Winner.white:
        white_win = 1
    elif env.winner == Winner.black:
        white_win = -1
    else:
        white_win = 0

    white.finish_game(white_win)
    black.finish_game(-white_win)

    data = []
    for i in range(len(white.moves)):
        data.append(white.moves[i])
        if i < len(black.moves):
            data.append(black.moves[i])

    return env, data
