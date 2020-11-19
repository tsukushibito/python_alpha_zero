from copy import copy
from abc import ABC, abstractmethod
from .game_state import GameState
from .action import Action


class Game(ABC):
    def __init__(self, initial_state: GameState):
        self._initial_state = initial_state
        self._state = initial_state

    @property
    def state(self):
        return self._state

    def step(self, action: Action) -> GameState:
        new_state = self._state.take_action(action)
        if new_state == None:
            return None

        self._state = new_state
        return self._state

    def reset(self, state: GameState) -> GameState:
        self._state = self._initial_state
        return self._state
