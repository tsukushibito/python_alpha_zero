from copy import copy
from .game_state import GameState
from .action import Action
from .player import Player


class Game():
    def __init__(self,
                 initial_state: GameState,
                 player0: Player,
                 player1: Player):
        self._initial_state = initial_state
        self._state = initial_state
        self._player0 = player0
        self._player1 = player1

    @property
    def state(self):
        return self._state

    @property
    def is_end(self):
        return self._state.is_end

    def step(self, action: Action) -> GameState:
        current_player = self._player0 if self._state.current_player == 0 else self._player1
        current_player.take_action(self._state)
        new_state = self._state.apply_action(action)
        if new_state == None:
            return None

        self._state = new_state
        return self._state

    def reset(self, state: GameState) -> GameState:
        self._state = self._initial_state
        return self._state
