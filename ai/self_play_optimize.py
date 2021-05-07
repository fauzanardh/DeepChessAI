import os
from collections import deque
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path
from random import shuffle
from threading import Thread

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from agent.model import ChessModel
from agent.player import ChessPlayer
from config import Config
from env.chess_env import ChessEnv, Winner, canon_input_planes, is_white_turn, evaluate
from util.data_helper import write_data, load_data, get_game_data_filenames


def start(config: Config):
    return SelfPlay(config).start()


class SelfPlay(object):
    def __init__(self, config: Config):
        self.config = config
        self.agent = None
        self.manager = Manager()
        self.cur_pipes = None
        self.buffer = []
        self.dataset = deque(), deque(), deque()
        self.filenames = None

    def load_model(self):
        agent = ChessModel(self.config)
        agent.load_latest()
        return agent

    def start(self):
        run = 3
        while True:
            print('=' * 32, "SelfPlay started!", '=' * 32)
            self.start_self_play()
            self.agent.reset()
            print('=' * 32, "Training started!", '=' * 32)
            self.start_training()
            self.agent.reset()
            self.config.play_path.replace(str(run), str(run + 1))
            run += 1

    def start_self_play(self):
        self.agent = self.load_model()
        self.cur_pipes = self.manager.list(
            [self.agent.get_pipes(self.config.play.search_threads) for _ in range(self.config.play.max_processes)]
        )
        self.buffer = []
        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
            for _ in range(self.config.play.max_game_per_file):
                futures.append(executor.submit(self_play_buffer, self.config, self.cur_pipes))

            game_idx = 0
            for future in as_completed(futures):
                game_idx += 1
                env, data = future.result()
                print(
                    f"game {game_idx:05} "
                    f"halfmoves={env.num_halfmoves:03} {env.winner:12} "
                    f"{'by resign' if env.is_resigned else ''} "
                    f"| fen={env.board.fen()}"
                )
            # game_idx = 0
            # while True:
            #     game_idx += 1
            #     env, data = futures.popleft().result()
            #     print(
            #         f"game {game_idx:05} "
            #         f"halfmoves={env.num_halfmoves:03} {env.winner:12} "
            #         f"{'by resign' if env.is_resigned else ''} "
            #         f"| fen={env.board.fen()}"
            #     )
            #     self.buffer += data
            #     if game_idx % self.config.play.max_game_per_file == 0:
            #         self.flush_buffer()
            #         break
            #     futures.append(executor.submit(self_play_buffer, self.config, self.cur_pipes))
        self.flush_buffer()

    def start_training(self):
        self.agent = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        self.filenames = deque(get_game_data_filenames(self.config))
        shuffle(self.filenames)
        total_steps = self.config.training.start_total_steps

        self.fill_queue()
        steps = self.train_epoch(self.config.training.epoch_to_checkpoint)
        total_steps += steps
        self.agent.save()
        state_arr, policy_arr, value_arr = self.dataset
        while len(state_arr) > self.config.training.dataset_size // 2:
            state_arr.popleft()
            policy_arr.popleft()
            value_arr.popleft()

    def train_epoch(self, epochs):
        ct = self.config.training
        state_arr, policy_arr, value_arr = self.collect_loaded_data()
        es_callback = EarlyStopping(monitor="val_loss", patience=6)
        self.agent.model.fit(
            state_arr, [policy_arr, value_arr],
            batch_size=ct.batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=0.2,
            callbacks=[es_callback]
        )
        steps = (state_arr.shape[0] // ct.batch_size) * epochs
        return steps

    def collect_loaded_data(self):
        state_arr, policy_arr, value_arr = self.dataset
        return np.asarray(state_arr, dtype=np.float32), \
               np.asarray(policy_arr, dtype=np.float32), \
               np.asarray(value_arr, dtype=np.float32)

    def fill_queue(self):
        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.training.max_processes) as executor:
            for _ in range(self.config.training.max_processes):
                if len(self.filenames) == 0:
                    break
                filename = str(self.filenames.popleft())
                print(f"Loading data from {filename}")
                futures.append(executor.submit(load_game_data, filename))
            while futures and len(self.dataset[0]) < self.config.training.dataset_size:
                for x, y in zip(self.dataset, futures.popleft().result()):
                    x.extend(y)
                if len(self.filenames) > 0:
                    filename = self.filenames.popleft()
                    print(f"Loading data from {filename}")
                    futures.append(executor.submit(load_game_data, filename))

    def compile_model(self):
        opt = Adam()
        losses = ["categorical_crossentropy", "mean_squared_error"]
        self.agent.model.compile(optimizer=opt, loss=losses, loss_weights=self.config.training.loss_weight)

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
        try:
            env.step(action)
        except Exception:
            env.adjudicate()

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


def load_game_data(filename):
    data = load_data(filename)
    return convert_data(data)


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
