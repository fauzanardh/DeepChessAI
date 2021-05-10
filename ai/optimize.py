import shutil
from pathlib import Path
from random import shuffle
from collections import deque
from concurrent.futures import ProcessPoolExecutor


from agent.model import ChessModel
from config import Config
from env.chess_env import canon_input_planes, is_white_turn, evaluate
from util.data_helper import load_data, get_game_data_filenames, get_tfr_filenames
from util.dataset_wrapper import DatasetWrapper

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping


def start(config: Config):
    return Optimizer(config).start()


class Optimizer(object):
    def __init__(self, config: Config):
        self.config = config
        self.agent = None
        self.dataset = deque(), deque(), deque()
        self.filenames = None
        self.executor = ProcessPoolExecutor(max_workers=self.config.training.max_processes)

    def load_model(self):
        agent = ChessModel(self.config)
        agent.load_latest()
        agent.model.summary()
        return agent

    def start(self):
        self.agent = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        self.filenames = deque(get_game_data_filenames(self.config))
        shuffle(self.filenames)
        # total_steps = self.config.training.start_total_steps

        # self.fill_queue()
        # steps = self.train_epoch(self.config.training.epoch_to_checkpoint)
        self.train_epoch()
        # total_steps += steps
        self.agent.save()
        state_arr, policy_arr, value_arr = self.dataset
        while len(state_arr) > self.config.training.dataset_size // 2:
            state_arr.popleft()
            policy_arr.popleft()
            value_arr.popleft()

    def get_epoch_size(self):
        files = get_tfr_filenames(self.config)
        epoch_size = 0
        for file in files:
            epoch_size += int(str(file).split('-')[1].split('.')[0])
        return epoch_size

    def train_epoch(self):
        ct = self.config.training
        epoch_size = self.get_epoch_size()
        train_size = int(0.9 * epoch_size)
        val_size = int(0.1 * epoch_size)
        dataset_wrapper = DatasetWrapper(self.config.tfr_path)
        state_arr, policy_arr, value_arr = self.collect_loaded_data()
        tb_callback = TensorBoard(log_dir="logs/", histogram_freq=1)
        # es_callback = EarlyStopping(monitor="val_loss", patience=3)
        self.agent.model.fit(
            dataset_wrapper.get_dataset(ct.batch_size, train_size, is_training=True),
            epochs=ct.epoch_to_checkpoint,
            steps_per_epoch=train_size // ct.batch_size,
            validation_data=dataset_wrapper.get_dataset(ct.batch_size, train_size),
            validation_steps=val_size // ct.batch_size,
            # callbacks=[tb_callback, es_callback]
            # callbacks=[tb_callback]
        )
        # self.agent.model.fit(
        #     state_arr, [policy_arr, value_arr],
        #     batch_size=ct.batch_size,
        #     epochs=epochs,
        #     shuffle=True,
        #     validation_split=0.2,
        #     callbacks=[tb_callback, es_callback]
        # )
        # steps = (state_arr.shape[0] // ct.batch_size) * epochs
        # return steps

    def collect_loaded_data(self):
        state_arr, policy_arr, value_arr = self.dataset
        return np.asarray(state_arr, dtype=np.float32), \
               np.asarray(policy_arr, dtype=np.float32), \
               np.asarray(value_arr, dtype=np.float32)

    def fill_queue(self):
        futures = deque()
        with self.executor as executor:
            for _ in range(self.config.training.max_processes):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.popleft()
                print(f"Loading data from {filename}")
                futures.append(executor.submit(load_game_data, self.config.play_path, filename))
            while futures and len(self.dataset[0]) < self.config.training.dataset_size:
                for x, y in zip(self.dataset, futures.popleft().result()):
                    x.extend(y)
                if len(self.filenames) > 0:
                    filename = self.filenames.popleft()
                    print(f"Loading data from {filename}")
                    futures.append(executor.submit(load_game_data, self.config.play_path, filename))

    def compile_model(self):
        lr_schedule = ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=10000,
            decay_rate=0.9
        )
        opt = Adam(learning_rate=lr_schedule)
        losses = ["categorical_crossentropy", "mean_squared_error"]
        self.agent.model.compile(optimizer=opt, loss=losses, loss_weights=self.config.training.loss_weight)


def load_game_data(play_path, filename: Path):
    data = load_data(filename)
    move_path = Path(play_path) / "done"
    move_path.mkdir(exist_ok=True)
    shutil.move(Path(play_path) / filename.name, move_path)
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
