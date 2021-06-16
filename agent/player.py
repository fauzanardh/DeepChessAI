from multiprocessing import Pipe
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import chess
import numpy as np

from config import Config
from env.chess_env import ChessEnv, Winner


class ActionStats(object):
    """
    This class holds the stats needed for AlphaZero Monte Carlo Tree Search (MCTS)
    for a specific action taken from a specific state

    :ivar n:
        Number of visits to this action by the algorithm
    :ivar w:
        Every time a child of this action is visited by the algorithm,
        this accumulates the value of that child. This is modified
        by a virtual loss which encourages threads to explore different nodes.
    :ivar q:
        Mean action value
    :ivar p:
        Prior probability of taking this action, given by the policy network output
    """

    def __init__(self) -> None:
        self.n = 0
        self.w = 0.0
        self.q = 0.0
        self.p = 0.0


class VisitStats(object):
    """
    This class holds the information to be used by the AlphaZero MCTS algorithm

    :ivar a:
        Holds the ActionStats instance
    :ivar sum_n:
        The sum of n inside the ActionStats dict
    """

    def __init__(self) -> None:
        self.a = defaultdict(ActionStats)
        self.sum_n = 0


class ChessPlayer(object):
    """
    Used to play the actual game of chess,
    choosing moves based on the policy and value network

    :ivar moves:
        Stores the info on the moves that have been performed during the game
    :ivar tree:
        Holds all of the visited game states and actions
    :ivar config:
        Stores the whole config for how to run
    :ivar n_labels:
        The number of possible move labels (a1b1, a1c1, ...)
    :ivar labels:
        All possible moves labels
    :ivar move_lookup:
        Lookup table for changing label to the index of that label in self.labels
    :ivar pipe_pool:
        The pipes to send the observation data of the game to the API
        and to get back the value and policy predicion from the network
    :ivar node_lock:
        A lookup table for changing the FEN game state to a Lock,
        used for indicating whether that state is currently being explored by another thread
    """

    def __init__(self, config: Config, pipes: Pipe = None, dummy: bool = False) -> None:
        self.moves = []
        self.tree = defaultdict(VisitStats)
        self.config = config
        self.n_labels = self.config.n_labels
        self.labels = self.config.labels
        self.move_lookup = {chess.Move.from_uci(move): i for move, i in zip(self.labels, range(self.n_labels))}
        if dummy:
            return
        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)

    def reset(self) -> None:
        """
        Reset the tree to begin a exploration
        """
        self.tree = defaultdict(VisitStats)

    def action(self, env: ChessEnv, can_stop: bool = True) -> Union[str, None]:
        """
        Used to figure out the next best move
        within the specified environment

        :param env:
            Environment in which the AI need to figure out the action from
        :param can_stop:
            Whether the AI is allowed to take no action (returns None)
        :return:
            Returns a string indicating the action to take in uci format
        """
        self.reset()

        root_value, naked_value = self.search_moves(env)
        policy = self.calc_policy(env)
        cur_action = int(np.random.choice(range(self.n_labels), p=self.apply_temperature(policy, env.num_halfmoves)))

        if can_stop and self.config.play.resign_threshold is not None and \
                root_value <= self.config.play.resign_threshold and \
                env.num_halfmoves > self.config.play.min_resign_turn:
            return None
        else:
            self.moves.append([env.observation, list(policy)])
            return self.config.labels[cur_action]

    def sl_action(self, observation: str, action: str, weight: float = 1) -> str:
        """
        Logs the action in self.moves.
        Used for generating game data from pgn file

        :param observation:
            Environment to use
        :param action:
            UCI format action to take
        :param weight:
            Weight to assign the taken action
        :return:
            The action, unmodified
        """
        policy = np.zeros(self.n_labels)
        k = self.move_lookup[chess.Move.from_uci(action)]
        policy[k] = weight
        self.moves.append([observation, list(policy)])
        return action

    def search_moves(self, env: ChessEnv) -> (float, float):
        """
        Looks at all the possible moves using the AlphaZero MCTS algorithm
        and find the highest move value possible

        :param env:
            Environment to search the moves from
        :return:
            The maximum value of all values predicted by each thread,
            and the first value that was predicted
        """
        futures = []
        with ThreadPoolExecutor(max_workers=self.config.play.search_threads) as executor:
            for _ in range(self.config.play.simulation_num_per_move):
                futures.append(executor.submit(self.search_my_move, env=env.copy(), is_root_node=True))

        values = [f.result() for f in futures]
        return np.max(values), values[0]

    def search_my_move(self, env: ChessEnv, is_root_node: bool = False) -> float:
        """
        Search for possible moves, add them to a tree search,
        and eventually returns the best move

        :param env:
            Environment to search the moves from
        :param is_root_node:
            Whether this is the root node of the search or not.
        :return:
            Value of the move calculated by the AlphaZero MCTS
        """
        if env.done:
            if env.winner == Winner.draw:
                return 0
            return -1

        state = state_key(env)
        with self.node_lock[state]:
            if state not in self.tree:
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[state].p = leaf_p
                return leaf_v

            # select step
            action_t = self.select_action_q_and_u(env, is_root_node)

            virtual_loss = self.config.play.virtual_loss

            cur_visit_stats = self.tree[state]
            cur_stats = cur_visit_stats.a[action_t]
            cur_visit_stats.sum_n += virtual_loss
            cur_stats.n += virtual_loss
            cur_stats.w += -virtual_loss
            cur_stats.q = cur_stats.w / cur_stats.n

        env.step(action_t.uci())
        leaf_v = self.search_my_move(env)
        leaf_v = -leaf_v

        # backup step
        with self.node_lock[state]:
            cur_visit_stats.sum_n += -virtual_loss + 1
            cur_stats.n += -virtual_loss + 1
            cur_stats.w += virtual_loss + leaf_v
            cur_stats.q = cur_stats.w / cur_stats.n

        return leaf_v

    def expand_and_evaluate(self, env: ChessEnv):
        """
        Expand a new leaf, and gets a prediction from policy and value network

        :param env:
            Environment to search the moves from
        :return:
            The policy and value predictions for this state
        """
        state_planes = env.canonical_input_planes()
        leaf_p, leaf_v = self.predict(state_planes)
        if not env.white_to_move:
            leaf_p = Config.flip_policy(leaf_p)
        return leaf_p, leaf_v

    def predict(self, state_planes) -> ():
        """
        Gets a prediction from the policy and value network,
        by sending it through the pipe to the prediction worker thread on another process

        :param state_planes:
            The observation state represented as planes
        :return:
            The policy and value predictions for the states
        """
        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        policy, value = pipe.recv()
        self.pipe_pool.append(pipe)
        return policy, value

    def select_action_q_and_u(self, env: ChessEnv, is_root_node: bool) -> chess.Move:
        """
        Picks the next action to explore using the AlphaZero MCTS Algorithm

        :param env:
            Environment to search the moves from
        :param is_root_node:
            Whether this is the root node of the MCTS search
        :return:
            The move to explore
        """
        state = state_key(env)
        cur_visit_stats = self.tree[state]

        if cur_visit_stats.p is not None:
            total_p = 1e-8
            for mov in env.board.legal_moves:
                mov_p = cur_visit_stats.p[self.move_lookup[mov]]
                cur_visit_stats.a[mov].p = mov_p
                total_p += mov_p
            for a_s in cur_visit_stats.a.values():
                a_s.p /= total_p
            cur_visit_stats.p = None

        _xx = np.sqrt(cur_visit_stats.sum_n + 1)

        e = self.config.play.noise_eps
        c_puct = self.config.play.c_puct
        dir_alpha = self.config.play.dirichlet_alpha

        best_s = -999
        best_a = None
        noise = None
        if is_root_node:
            noise = np.random.dirichlet([dir_alpha] * len(cur_visit_stats.a))

        i = 0
        for action, a_s in cur_visit_stats.a.items():
            _p = a_s.p
            if is_root_node:
                _p = (1 - e) * _p + e * noise[i]
                i += 1
            b = a_s.q + c_puct * _p * _xx / (1 + a_s.n)
            # print(a_s.q, c_puct, _p, _xx, a_s.n)
            if b > best_s:
                best_s = b
                best_a = action

        return best_a

    def apply_temperature(self, policy, turn: int):
        """
        Applies a random fluctuation to the probability of choosing various actions

        :param policy:
            The list of probabilities of taking each action
        :param turn:
            Number of turns that have occured in the game so far
        :return:
            The policy with randomly perturbed based on the temperature
        """
        tau = np.power(self.config.play.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.n_labels)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1 / tau)
            ret /= np.sum(ret)
            return ret

    def calc_policy(self, env: ChessEnv):
        """
        calc π(a|s0)

        :param env:
            Environment to search the moves from
        :return:
            A list of probabilities of taking each action, calculated based on visit counts
        """
        state = state_key(env)
        cur_visit_stats = self.tree[state]
        policy = np.zeros(self.n_labels)
        for action, a_s in cur_visit_stats.a.items():
            policy[self.move_lookup[action]] = a_s.n

        policy /= np.sum(policy)
        return policy

    def finish_game(self, z) -> None:
        """
        When game is finished, updates all values of the past moves based on the result

        :param z:
            1 == win, -1 == lose, 0 == draw
        """
        for move in self.moves:
            move += [z]


def state_key(env: ChessEnv) -> str:
    """
    :param env:
        Environment to decode
    :return:
        A string representation of the game state
    """
    fen = env.observation.rsplit(' ', 1)
    return fen[0]
