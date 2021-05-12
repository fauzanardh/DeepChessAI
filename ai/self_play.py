import os
from pathlib import Path
from threading import Thread
from collections import deque
from concurrent.futures.process import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import Manager

from config import Config
from agent.model import ChessModel
from agent.player import ChessPlayer
from env.chess_env import ChessEnv, Winner
from util.data_helper import write_data


def start(config: Config):
    return SelfPlay(config).start()


class SelfPlay(object):
    def __init__(self, config: Config):
        self.config = config
        self.agent = self.load_model()
        self.manager = Manager()
        self.cur_pipes = self.manager.list(
            [self.agent.get_pipes(self.config.play.search_threads) for _ in range(self.config.play.max_processes)]
        )
        self.buffer = []

    def load_model(self):
        agent = ChessModel(self.config)
        agent.load_latest()
        agent.model.summary()
        return agent

    def start(self):
        self.buffer = []
        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
            for _ in range(self.config.play.max_processes * 2):
                futures.append(executor.submit(self_play_buffer, self.config, self.cur_pipes))

            game_idx = 0
            while True:
                game_idx += 1
                env, data = futures.popleft().result()
                print(
                    f"game {game_idx:05} "
                    f"halfmoves={env.num_halfmoves:03} {env.winner:12} "
                    f"{'by resign' if env.is_resigned else ''} "
                    f"| fen={env.observation}"
                )
                self.buffer += data
                if game_idx % self.config.play.max_game_per_file == 0:
                    self.flush_buffer()
                futures.append(executor.submit(self_play_buffer, self.config, self.cur_pipes))

    def flush_buffer(self):
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(self.config.play_path, f"play_{game_id}.json")
        Path(self.config.play_path).mkdir(exist_ok=True)
        thread = Thread(target=write_data, args=(path, self.buffer))
        thread.start()
        self.buffer = []


def self_play_buffer(config: Config, pipes):
    env = ChessEnv().reset()
    cur_pipe = pipes.pop()

    white = ChessPlayer(config, pipes=cur_pipe)
    black = ChessPlayer(config, pipes=cur_pipe)

    while not env.done:
        if env.white_to_move:
            action = white.action(env)
        else:
            action = black.action(env)
        print('=' * 20)
        print(action)
        env.step(action)
        print('-' * 20)
        print(env.board)
        print('=' * 20)

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

    pipes.append(cur_pipe)
    return env, data
