import itertools
from copy import deepcopy
from enum import Enum
from typing import Tuple, List
from dataclasses import dataclass
from dataclasses import field
from ..game import GameState
from ..game import Action


class Dir(Enum):
    UP_LEFT = (-1, -1)
    UP = (0, -1)
    UP_RIGHT = (1, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    DOWN_LEFT = (-1, 1)
    DOWN = (0, 1)
    DOWN_RIGHT = (1, 1)


class Square(Enum):
    BLACK = 0
    WHITE = 1
    EMPTY = 2


def create_initial_board(board_size: int) -> List[List[Square]]:
    board = [[Square.EMPTY] * board_size for _ in range(board_size)]

    p = (board_size // 2) - 1
    board[p][p] = board[p + 1][p + 1] = Square.BLACK
    board[p][p + 1] = board[p + 1][p] = Square.WHITE

    return board


@dataclass(frozen=True)
class ReversiAction(Action):
    _row: int = 0
    _col: int = 0
    _pass: bool = False

    @property
    def value(self):
        return ReversiState.board_size * self._row + self._col \
            if not self._pass \
            else ReversiState.board_size * ReversiState.board_size

    @property
    def pos(self):
        return (self._row, self._col) \
            if not self._pass \
            else(-1, -1)

    def is_pass(self):
        return self._pass


@dataclass(frozen=True)
class ReversiState(GameState):
    _depth: int = 0
    _board: List[List[Square]] = field(
        default_factory=lambda: create_initial_board(ReversiState.board_size))
    _is_end: bool = False

    board_size: int = 8

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def current_player(self) -> int:
        return self._depth % 2

    @property
    def player0_board(self) -> List[int]:
        return [(1 if i == Square.BLACK else 0) for i in itertools.chain.from_iterable(self._board)]

    @property
    def player1_board(self) -> List[int]:
        return [(1 if i == Square.WHITE else 0) for i in itertools.chain.from_iterable(self._board)]

    @property
    def is_end(self) -> bool:
        return self._is_end

    @property
    def is_current_player_winner(self) -> bool:
        if not self.is_end():
            return False
        player0_score = self.player0_board.count(1)
        player1_score = self.player1_board.count(1)
        return player0_score > player1_score \
            if self.current_player == 0 \
            else player1_score > player0_score

    @property
    def is_draw(self) -> bool:
        if not self.is_end():
            return False
        return self.player0_board.count(1) == self.player1_board.count(1)

    @property
    def is_current_player_loser(self) -> bool:
        if not self.is_end():
            return False

        return not self.is_draw and not self.is_current_player_winner

    @property
    def allowed_actions(self) -> List[Action]:
        actions = []
        for r in range(8):
            for c in range(8):
                action = ReversiAction(r, c)
                if self._is_allowed_action(action):
                    actions.append(action)

        return actions

    def take_action(self, action: Action) -> GameState:
        assert type(action) is ReversiAction

        if self._is_end:
            # 終了したゲームなのでアクションは受け付けない
            return None

        if action.is_pass() and len(self.allowed_actions) > 0:
            # 合法手があるにも関わらずパスすることはできない
            return None

        flip_dirs = []
        board = deepcopy(self._board)
        if not action.is_pass() \
                and self._is_allowed_action(action, flip_dirs):
            r, c = action.pos
            board[r][c] = self.player_square
            for dir in flip_dirs:
                d = dir.value
                r_it = r + d[1]
                c_it = c + d[0]
                while board[r_it][c_it] == self.opposing_square:
                    board[r_it][c_it] = self.player_square
                    r_it += d[1]
                    c_it += d[0]

        next_state = ReversiState(self.depth + 1, board)
        if action.is_pass and len(next_state.allowed_actions) == 0:
            next_state = ReversiState(next_state.depth, board, True)

        return next_state

    @property
    def player_square(self) -> Square:
        return Square.BLACK if self.current_player == 0 else Square.WHITE

    @property
    def opposing_square(self) -> Square:
        return Square.WHITE if self.current_player == 0 else Square.BLACK

    def _is_valid_position(self, row: int, col: int) -> bool:
        return row >= 0 and row < ReversiState.board_size \
            and col >= 0 and col < ReversiState.board_size

    def _distance(self, r0: int, c0: int, r1: int, c1: int) -> int:
        dr = abs(r0 - r1)
        dc = abs(c0 - c1)
        return dr if dr > 0 else dc

    def _is_allowed_action(self, action: Action, out_flip_dirs: list = None) -> bool:
        assert type(action) is ReversiAction

        r, c = action.pos
        if not self._is_valid_position(r, c):
            return False

        if self._board[r][c] != Square.EMPTY:
            return False

        is_arrowed = False
        for dir in Dir:
            d = dir.value

            r_it = r + d[1]
            c_it = c + d[0]

            while self._is_valid_position(r_it, c_it) \
                    and self._board[r_it][c_it] == self.opposing_square:
                r_it += d[1]
                c_it += d[0]

            if self._is_valid_position(r_it, c_it) \
                    and self._distance(r, c, r_it, c_it) > 1 \
                    and self._board[r_it][c_it] == self.player_square:
                is_arrowed = True

                if out_flip_dirs != None:
                    out_flip_dirs.append(dir)

        return is_arrowed
