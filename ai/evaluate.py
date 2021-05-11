from collections import deque
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed

from agent.model import ChessModel
from agent.player import ChessPlayer
from config import Config
from env.chess_env import ChessEnv, Winner


def start(config: Config):
    return Optimizer(config).start()


class Optimizer(object):
    def __init__(self, config: Config):
        self.config = config
        self.init_eval_config()
        self.dataset = deque(), deque(), deque()
        self.filenames = None
        self.manager = Manager()
        self.cur_model = None
        self.cur_pipes = []
        self.new_model = None
        self.new_pipes = []
        self.executor = ProcessPoolExecutor(max_workers=self.config.evaluate.max_processes)

    def init_eval_config(self):
        self.config.play.c_puct = self.config.evaluate.c_puct
        self.config.play.game_num = self.config.evaluate.game_num
        self.config.play.noise_eps = self.config.evaluate.noise_eps
        self.config.play.simulation_num_per_move = self.config.evaluate.simulation_num_per_move
        self.config.play.tau_decay_rate = self.config.evaluate.tau_decay_rate

    def start(self):
        self.cur_model = ChessModel(self.config)
        model_weights = self.cur_model.get_weights_path()
        self.cur_model.load_path(model_weights[-2])
        self.cur_pipes = self.manager.list(
            [self.cur_model.get_pipes(self.config.play.search_threads) for _ in range(self.config.evaluate.max_processes)]
        )
        self.new_model = ChessModel(self.config)
        self.new_model.load_path(model_weights[-1])
        self.new_pipes = self.manager.list(
            [self.new_model.get_pipes(self.config.play.search_threads) for _ in range(self.config.evaluate.max_processes)]
        )
        is_better = self.evaluate_model()
        if is_better:
            print(f"Model {model_weights[-1]} is better than {model_weights[-2]}!")
        else:
            print(f"Model {model_weights[-2]} is better than {model_weights[-1]}!")

    def evaluate_model(self):
        futures = []
        with self.executor as executor:
            for game_idx in range(self.config.evaluate.game_num):
                future = executor.submit(
                    play_game,
                    self.config,
                    cur=self.cur_pipes,
                    new=self.new_pipes,
                    cur_white=(game_idx % 2 == 0)
                )
                futures.append(future)
            results = []
            for future in as_completed(futures):
                new_score, env, cur_white = future.result()
                results.append(new_score)
                win_rate = sum(results) / len(results)
                game_idx = len(results)
                print(
                    f"game {game_idx:05} new_score{new_score:.2f} as {'black' if cur_white else 'white'} "
                    f"{'by resign' if env.is_resigned else ''} "
                    f"win_rate={win_rate*100:5.2f} "
                    f"| fen={env.observation}"
                )
                if len(results) - sum(results) >= self.config.evaluate.game_num * (1 - self.config.evaluate.replace_rate):
                    print(f"lose count reached {results.count(0)}, worse model.")
                    return False
                if sum(results) >= self.config.evaluate.game_num * self.config.evaluate.replace_rate:
                    print(f"win count reached {results.count(1)}, better model.")
                    return True

            win_rate = sum(results) / len(results)
            print(f"win_rate={win_rate*100:.2f}%")
            return win_rate >= self.config.evaluate.replace_rate


def play_game(config: Config, cur, new, cur_white: bool):
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
