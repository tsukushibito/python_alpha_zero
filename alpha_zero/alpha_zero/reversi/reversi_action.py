from ..game import Action
from .reversi_constants import REVERSI_BOARD_SIZE


class ReversiAction(Action):
    def __init__(self, row: int, col: int, is_pass: bool = False):
        self._row: int = row
        self._col: int = col
        self._pass: bool = is_pass

    @property
    def value(self) -> int:
        return REVERSI_BOARD_SIZE * self._row + self._col \
            if not self._pass \
            else REVERSI_BOARD_SIZE * REVERSI_BOARD_SIZE

    @property
    def pos(self) -> (int, int):
        return (self._row, self._col) \
            if not self._pass \
            else(-1, -1)

    @property
    def is_pass(self) -> bool:
        return self._pass
