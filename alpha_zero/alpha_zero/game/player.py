from abc import ABC, abstractmethod
from typing import Any
from .action import Action
from .game_state import GameState


class Player(ABC):
    @abstractmethod
    def take_action(self, state: GameState) -> Action:
        pass
