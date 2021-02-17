from .reversi_action import ReversiAction
from .reversi_state import ReversiState


class ReversiAiPlayer:
    def take_action(self, state: ReversiState) -> ReversiAction:
        return ReversiAction(0, 0)
