import copy
from enum import Enum

import chess.pgn
import numpy as np

PIECES_ORDER = "KQRBNPkqrbnp"  # 12x8x8
PIECES_ORDER_INDEX = {PIECES_ORDER[i]: i for i in range(len(PIECES_ORDER))}
CASTLING_ORDER = "KQkq"  # 4x8x8

Winner = Enum("Winner", "white black draw")


class ChessEnv:
    def __init__(self):
        self.board = None
        self.num_halfmoves = 0
        self.winner = None
        self.is_resigned = False
        self.result = None

    # Resets the environment to begin a new game
    def reset(self):
        self.board = chess.Board()
        self.num_halfmoves = 0
        self.winner = None
        self.is_resigned = False
        return self

    # This function acts like reset,
    # but it resets to the supplied board position
    def update(self, board):
        self.board = chess.Board(board)
        self.winner = None
        self.is_resigned = False
        return self

    @property
    def done(self):
        return self.winner is not None

    @property
    def white_won(self):
        return self.winner == Winner.white

    @property
    def white_to_move(self):
        return self.board.turn == chess.WHITE

    # Takes an action and update the game state
    def step(self, action: str, check_over=True):
        # check if the action returned from the AI is NONE
        # meaning it don't have any moves
        if check_over and action is None:
            self._resign()

        self.board.push_uci(action)
        self.num_halfmoves += 1

        if check_over and self.board.result(claim_draw=True) != '*':
            self._game_over()

    def _resign(self):
        self.is_resigned = True
        if self.white_to_move:
            self.winner = Winner.black
            self.result = "0-1"
        else:
            self.winner = Winner.white
            self.result = "1-0"

    def _game_over(self):
        if self.winner is None:
            self.result = self.board.result(claim_draw=True)
            if self.result == "1-0":
                self.winner = Winner.white
            elif self.result == "0-1":
                self.winner = Winner.black
            else:
                self.winner = Winner.draw

    def adjudicate(self):
        score = self._evaluate(absolute=True)
        # if the score is less than 0.01 it's pretty much a draw,
        # since the player advantage is not significant
        if abs(score) < 0.01:
            self.winner = Winner.draw
            self.result = "1/2-1/2"
        elif score > 0:
            self.winner = Winner.white
            self.result = "1-0"
        else:
            self.winner = Winner.black
            self.result = "0-1"

    def copy(self):
        env = copy.copy(self)
        env.board = copy.copy(self.board)
        return env

    def render(self):
        print(self.board)

    @property
    def observation(self):
        return self.board.fen()

    # Returns a representation of the board using an (18, 8, 8) dimension array
    def canonical_input_planes(self):
        return canon_input_planes(self.board.fen())

    # try to evaluate the current board value
    # based on the pieces on the board
    def _evaluate(self, absolute=False):
        pieces_val = {'K': 3, 'Q': 14, 'R': 5, 'B': 3.25, 'N': 3, 'P': 1}
        ans = 0.0
        total = 0
        for c in self.board.fen().split(' ')[0]:
            if not c.isalpha():
                continue

            if c.isupper():
                ans += pieces_val[c]
                total += pieces_val[c]
            else:
                ans -= pieces_val[c.upper()]
                total += pieces_val[c.upper()]
        v = ans/total
        if not absolute and not is_white_turn(self.board.fen()):
            v = -v
        assert abs(v) < 1, "Something went wrong when calculating the pieces value"
        return np.tanh(v * 3)


def flip_fen(fen):
    _fen = fen.split(' ')
    rows = _fen[0].split('/')

    def swap_case(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a

    def swap_all(aa):
        return "".join([swap_case(a) for a in aa])

    return "/".join([swap_all(row) for row in reversed(rows)]) \
           + " " + ('w' if _fen[1] == 'b' else 'b') \
           + " " + "".join(sorted(swap_all(_fen[2]))) \
           + " " + _fen[3] + " " + _fen[4] + " " + _fen[5]


def alg_to_coord(alg):
    # example:
    # move = "a1", this will return (0, 0)
    # move = "h8", this will return (7, 7)
    rank = 8 - int(alg[1])
    file = ord(alg[0]) - ord('a')
    return rank, file


def coord_to_alg(coord):
    # the complete opposite to function `alg_to_coord`
    letter = chr(ord('a') + coord[1])
    number = str(8 - coord[0])
    return letter + number


def aux_planes(fen):
    _fen = fen.split(' ')

    # creating array for storing en-passant move
    en_passant = np.zeros((8, 8), dtype=np.float32)
    if _fen[3] != '-':
        rank, file = alg_to_coord(_fen[3])
        en_passant[rank][file] = 1

    # fifty move rule
    fifty_move_count = int(_fen[4])
    fifty_move = np.full((8, 8), fifty_move_count, dtype=np.float32)

    castling = _fen[2]
    auxiliary_planes = [
        np.full((8, 8), int('K' in castling), dtype=np.float32),
        np.full((8, 8), int('Q' in castling), dtype=np.float32),
        np.full((8, 8), int('k' in castling), dtype=np.float32),
        np.full((8, 8), int('q' in castling), dtype=np.float32),
        fifty_move,
        en_passant
    ]
    ret = np.asarray(auxiliary_planes, dtype=np.float32)
    assert ret.shape == (6, 8, 8), "Wrong dimension for the auxiliary planes!"
    return ret


def replace_tags(fen):
    fen = fen.split(" ")[0]
    fen = fen.replace("2", "11")
    fen = fen.replace("3", "111")
    fen = fen.replace("4", "1111")
    fen = fen.replace("5", "11111")
    fen = fen.replace("6", "111111")
    fen = fen.replace("7", "1111111")
    fen = fen.replace("8", "11111111")
    return fen.replace("/", "")


def to_planes(fen):
    board_state = replace_tags(fen)
    pieces_both = np.zeros((12, 8, 8), dtype=np.float32)
    for rank in range(8):
        for file in range(8):
            v = board_state[rank * 8 + file]
            if v.isalpha():
                pieces_both[PIECES_ORDER_INDEX[v]][rank][file] = 1
    assert pieces_both.shape == (12, 8, 8), "Wrong dimension for the pieces arrays!"
    return pieces_both


def all_input_planes(fen):
    current_aux_planes = aux_planes(fen)
    pieces_both_player = to_planes(fen)
    ret = np.vstack((pieces_both_player, current_aux_planes))
    assert ret.shape == (18, 8, 8), "Wrong dimension for the input planes!"
    return ret


def canon_input_planes(fen):
    if not is_white_turn(fen):
        fen = flip_fen(fen)
    return all_input_planes(fen)


def is_white_turn(fen):
    return fen.split(' ')[1] == 'w'

