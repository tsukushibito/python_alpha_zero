from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
from copy import copy
from .action import Action


@dataclass
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
    def player0_board(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def player1_board(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def is_end(self) -> bool:
        pass

    @property
    @abstractmethod
    def allowed_actions(self) -> list[Action]:
        pass

    @abstractmethod
    def take_action(self, action: Action):
        pass
