import copy
import shutil
from collections import deque
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from agent.model import ChessModel
from agent.player import ChessPlayer
from config import Config
from env.chess_env import ChessEnv, Winner
from util.data_helper import get_tfr_filenames
from util.dataset_tool import TFRecordExporter
from util.dataset_wrapper import DatasetWrapper


def start(config: Config):
    return SelfPlay(config).start()


class SelfPlay(object):
    def __init__(self, config: Config):
        self.config = config
        self.manager = Manager()

    def load_best(self, config=None):
        agent = ChessModel(self.config if config is None else config)
        agent.load_best()
        return agent

    def load_latest(self, config=None):
        agent = ChessModel(self.config if config is None else config)
        agent.load_latest()
        return agent

    def start(self):
        while True:
            print('=' * 32, "SelfPlay started!", '=' * 32)
            self.start_self_play()
            print('=' * 32, "Training started!", '=' * 32)
            self.start_training()
            print('=' * 32, "Evaluator started!", '=' * 32)
            self.start_evaluate()

    def start_self_play(self):
        best_model = self.load_best()
        cur_pipes = self.manager.list(
            [best_model.get_pipes(self.config.play.search_threads) for _ in range(self.config.play.max_processes)]
        )
        futures = deque()
        game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        exporter = TFRecordExporter(f"self_play_{game_id}-{self.config.play.max_game_per_file}", self.config)
        lock = self.manager.Lock()
        with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
            for _ in range(self.config.play.max_game_per_file):
                futures.append(executor.submit(self_play_buffer, self.config, cur_pipes))

            game_idx = 0
            for future in as_completed(futures):
                game_idx += 1
                env, data = future.result()
                exporter.add_from_buffer(data, lock)
                print(
                    f"game {game_idx:05} "
                    f"halfmoves={env.num_halfmoves:03} {env.winner:12} "
                    f"{'by resign' if env.is_resigned else ''} "
                    f"| fen={env.observation}"
                )
        exporter.close()

    def start_training(self):
        agent = self.load_best()
        self.training(agent)

    def training(self, agent):
        self.compile_model(agent)
        self.train_epoch(agent)
        agent.save()

    def get_epoch_size(self):
        files = get_tfr_filenames(self.config)
        epoch_size = 0
        for file in files:
            epoch_size += int(str(file).split('-')[1].split('.')[0])
        return epoch_size

    def train_epoch(self, agent):
        ct = self.config.training
        epoch_size = self.get_epoch_size()
        print(f"Epoch Size: {epoch_size}")
        train_size = int(0.9 * epoch_size)
        latest_file = list(Path(self.config.tfr_path).glob("*.tfrecords"))
        dataset_wrapper = DatasetWrapper(self.config, tfr_files=latest_file)
        agent.model.fit(
            dataset_wrapper.get_dataset(train_size),
            epochs=ct.epoch_to_checkpoint,
            validation_data=dataset_wrapper.get_dataset(train_size, is_training=False),
        )

    def compile_model(self, agent):
        lr_schedule = ExponentialDecay(
            initial_learning_rate=3e-4,
            decay_steps=10000,
            decay_rate=0.9
        )
        opt = Adam(learning_rate=lr_schedule)
        losses = ["categorical_crossentropy", "mean_squared_error"]
        agent.model.compile(optimizer=opt, loss=losses, loss_weights=self.config.training.loss_weight)

    def start_evaluate(self):
        config = copy.deepcopy(self.config)
        config.play.c_puct = self.config.evaluate.c_puct
        config.play.game_num = self.config.evaluate.game_num
        config.play.noise_eps = self.config.evaluate.noise_eps
        config.play.simulation_num_per_move = self.config.evaluate.simulation_num_per_move
        config.play.tau_decay_rate = self.config.evaluate.tau_decay_rate

        best_model = self.load_best(config)
        best_pipes = self.manager.list(
            [best_model.get_pipes(self.config.play.search_threads) for _ in range(self.config.evaluate.max_processes)]
        )
        latest_model = self.load_latest(config)
        latest_pipes = self.manager.list(
            [latest_model.get_pipes(self.config.play.search_threads) for _ in range(self.config.evaluate.max_processes)]
        )
        is_better = self.evaluate_model(best_pipes, latest_pipes)
        if is_better:
            path = Path(self.config.model_path) / "best_model"
            best_model_path = path / "model.h5"
            all_weights = list(Path(self.config.model_path).glob("model_*.h5"))
            latest_weight = str(max(all_weights, key=lambda x: int(str(x).split("_")[1].split(".")[0])))
            print("Latest model is better, copying the model to the best_model/ folder")
            shutil.copy(latest_weight, best_model_path)
            self.move_tfr()
        else:
            print("Latest model is worse, continuing")

    def evaluate_model(self, best_pipes, latest_pipes):
        futures = []
        with ProcessPoolExecutor(max_workers=self.config.evaluate.max_processes) as executor:
            for game_idx in range(self.config.evaluate.game_num):
                future = executor.submit(
                    evaluate_buffer,
                    self.config,
                    cur=best_pipes,
                    new=latest_pipes,
                    cur_white=(game_idx % 2 == 0)
                )
                futures.append(future)
            res = []
            for future in as_completed(futures):
                latest_score, env, is_bm_white = future.result()
                res.append(latest_score)
                win_rate = sum(res) / len(res)
                game_idx = len(res)
                print(
                    f"game {game_idx:05} new_score{latest_score:.2f} as {'black' if is_bm_white else 'white'} "
                    f"{'by resign' if env.is_resigned else ''} "
                    f"win_rate={win_rate * 100:5.2f} "
                    f"| fen={env.observation}"
                )
                if len(res) - sum(res) >= self.config.evaluate.game_num * (1 - self.config.evaluate.replace_rate):
                    print(f"lose count reached {res.count(0)}, worse model.")
                    return False
                if sum(res) >= self.config.evaluate.game_num * self.config.evaluate.replace_rate:
                    print(f"win count reached {res.count(1)}, better model.")
                    return True

            win_rate = sum(res) / len(res)
            print(f"win_rate={win_rate * 100:.2f}%")
            return win_rate >= self.config.evaluate.replace_rate

    def move_tfr(self):
        tfr_path = Path(self.config.tfr_path)
        finish_path = tfr_path / "done"
        finish_path.mkdir(exist_ok=True)
        tfrs = tfr_path.glob("*.tfrecords")
        for tfr in tfrs:
            shutil.move(tfr, finish_path)


def self_play_buffer(config: Config, pipes):
    env = ChessEnv().reset()
    cur_pipe = pipes.pop()

    white = ChessPlayer(config, pipes=cur_pipe)
    black = ChessPlayer(config, pipes=cur_pipe)

    while not env.done:
        if env.white_to_move:
            action = white.action(env, can_stop=False)
        else:
            action = black.action(env, can_stop=False)
        try:
            env.step(action)
        except Exception as e:
            print(e)
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


def evaluate_buffer(config: Config, cur, new, cur_white: bool):
    cur_pipe = cur.pop()
    new_pipe = new.pop()
    env = ChessEnv().reset()

    cur_player = ChessPlayer(config, pipes=cur_pipe)
    new_player = ChessPlayer(config, pipes=new_pipe)
    print(f"cur_white={cur_white}")
    if cur_white:
        white, black = cur_player, new_player
        print("Playing as black")
    else:
        white, black = new_player, cur_player
        print("Playing as white")

    while not env.done:
        if env.white_to_move:
            action = white.action(env)
        else:
            action = black.action(env)
        try:
            env.step(action)
            # print('=' * 20)
            # print(env.board)
            # print('=' * 20)
        except Exception as e:
            print(e)
            env.adjudicate()

        if env.num_halfmoves >= config.play.max_game_length:
            env.adjudicate()

    if env.winner == Winner.draw:
        new_score = 0.5
    elif env.white_won == cur_white:
        new_score = 0
    else:
        new_score = 1

    cur.append(cur_pipe)
    new.append(new_pipe)
    return new_score, env, cur_white
