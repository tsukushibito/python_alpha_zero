from abc import ABC, abstractmethod
from typing import Any, List
from copy import copy
from .action import Action


class GameState(ABC):
    @property
    @abstractmethod
    def depth(self) -> int:
        pass

    @property
    @abstractmethod
    def current_player(self) -> int:
        pass

    @property
    @abstractmethod
    def player0_board(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def player1_board(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def is_end(self) -> bool:
        pass

    @property
    def is_current_player_winner(self) -> bool:
        pass

    @property
    def is_draw(self) -> bool:
        pass

    @property
    def is_current_player_loser(self) -> bool:
        pass

    @property
    @abstractmethod
    def allowed_actions(self) -> List[Action]:
        pass

    @abstractmethod
    def take_action(self, action: Action):
        pass
