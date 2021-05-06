from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import chess
import numpy as np

from config import Config
from env.chess_env import ChessEnv, Winner


# This class holds the stats needed for AlphaZero Monte Carlo Tree Search (MCTS)
# for a specific action taken from a specific state
class ActionStats(object):
    # n: number of visits to this action by the algorithm
    # w: every time a child of this action is visited by the algorithm,
    #    this accumulates the value of that child. This is modified
    #    by a virtual loss which encourages threads to explore different nodes.
    # q: mean action value
    # p: prior probability of taking this action, given by the policy network output
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0


# This class holds the information
# to be used by the AlphaZero MCTS algorithm
class VisitStats(object):
    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_n = 0


class ChessPlayer(object):
    def __init__(self, config: Config, pipes=None):
        self.moves = []
        self.tree = defaultdict(VisitStats)
        self.config = config
        self.n_labels = self.config.n_labels
        self.labels = self.config.labels
        self.move_lookup = {chess.Move.from_uci(move): i for move, i in zip(self.labels, range(self.n_labels))}
        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)

    # Reset the tree to begin a exploration
    def reset(self):
        self.tree = defaultdict(VisitStats)

    # Figure out what is the best next move
    # within the specified environment
    def action(self, env: ChessEnv, can_stop=True):
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

    # Looks at all the possible moves using the AlphaZero MCTS algorithm
    # find the highest move value possible
    def search_moves(self, env: ChessEnv):
        futures = []
        with ThreadPoolExecutor(max_workers=self.config.play.search_threads) as executor:
            for _ in range(self.config.play.simulation_num_per_move):
                futures.append(executor.submit(self.search_my_move, env=env.copy(), is_root_node=True))

        values = [f.result() for f in futures]
        return np.max(values), values[0]

    # Search for possible moves, adds them to a tree search,
    # and eventually returns the best move
    def search_my_move(self, env: ChessEnv, is_root_node=False):
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
            cur_stats.w = virtual_loss + leaf_v
            cur_stats.q = cur_stats.w / cur_stats.n

        return leaf_v

    # Gets a prediction for the policy
    # and value of the state within the given env
    def expand_and_evaluate(self, env: ChessEnv):
        state_planes = env.canonical_input_planes()
        leaf_p, leaf_v = self.predict(state_planes)
        if not env.white_to_move:
            leaf_p = Config.flip_policy(leaf_p)
        return leaf_p, leaf_v

    # Gets a prediction from the policy and value network
    def predict(self, state_planes):
        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        policy, value = pipe.recv()
        self.pipe_pool.append(pipe)
        return policy, value
        # data = np.asarray([state_planes], dtype=np.float32)
        # policy, value = self.agent.model.predict_on_batch(data)
        # return policy[0], float(value[0])

    # Picks the next action to explore using the AlphaZero MCTS algorithm
    # The action are based on the action which maximize the maximum action value
    def select_action_q_and_u(self, env: ChessEnv, is_root_node):
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
            if b > best_s:
                best_s = b
                best_a = action

        return best_a

    # Applies a random fluctuation to probability of choosing various actions
    def apply_temperature(self, policy, turn):
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

    # calc π(a|s0)
    def calc_policy(self, env: ChessEnv):
        state = state_key(env)
        cur_visit_stats = self.tree[state]
        policy = np.zeros(self.n_labels)
        for action, a_s in cur_visit_stats.a.items():
            policy[self.move_lookup[action]] = a_s.n

        policy /= np.sum(policy)
        return policy

    # When game is done, updates the value of
    # all past moves based on the result
    def finish_game(self, z):
        for move in self.moves:
            move += [z]


def state_key(env: ChessEnv):
    fen = env.board.fen().rsplit(' ', 1)
    return fen[0]
