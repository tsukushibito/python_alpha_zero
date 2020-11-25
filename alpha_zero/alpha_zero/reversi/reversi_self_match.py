import os
import pickle
from typing import List
from datetime import datetime
from pathlib import Path
import numpy as np
from . import ReversiDualNetwork
from . import ReversiState
from . import search_with_mtcs, choice_next_action


RECORD_DATA_DIR = './record/'
RECORD_DATA_FILE_EXT = '.reversi.record'


def _self_match_impl(dual_network: ReversiDualNetwork, temperature: float, evaluation_count: int):
    record = []

    state = ReversiState()

    while not state.is_end:
        scores = search_with_mtcs(
            dual_network, state, temperature, evaluation_count)

        policies = [0] * ReversiDualNetwork.OUT_POLICY_COUNT
        for action, policy in zip(state.allowed_actions, scores):
            policies[action.value] = policy

        board = [state.player0_board, state.player1_board] \
            if state.current_player == 0 \
            else [state.player1_board, state.player0_board]
        record.append([board, policies, None])

        action = choice_next_action(state.allowed_actions, scores)
        state = state.take_action(action)

    value = 0
    if state.current_player == 0:
        if state.is_current_player_winner:
            value = 1
        elif state.is_draw:
            value = 0
        else:
            value = -1
    else:
        if state.is_current_player_winner:
            value = -1
        elif state.is_draw:
            value = 0
        else:
            value = 1

    for i in range(len(record)):
        record[i][2] = value
        value = -value

    return record


def save_self_match_record(record: List):
    now = datetime.now()
    os.makedirs(RECORD_DATA_DIR, exist_ok=True)
    name = '{:04}{:02}{:02}{:02}{:02}{:02}'.format(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second)
    path = RECORD_DATA_DIR + name + RECORD_DATA_FILE_EXT

    with open(path, mode='wb') as f:
        pickle.dump(record, f)


def load_last_self_match_record() -> List:
    path = sorted(Path(RECORD_DATA_DIR).glob('*'+RECORD_DATA_FILE_EXT))[-1]
    with path.open(mode='rb') as f:
        return pickle.load(f)


def record_to_model_fitting_data(record: List) -> (np.ndarray, List[np.ndarray]):
    states, policies, values = zip(*record)
    a, b, c = ReversiDualNetwork.INPUT_SHAPE
    xs = np.array(states)
    xs = xs.reshape((len(xs), c, a, b)).transpose(0, 2, 3, 1)
    y_policies = np.array(policies)
    y_values = np.array(values)
    return xs, [y_policies, y_values]


def run_self_match(game_count: int, temperature: float, evaluation_count: int) -> List:
    record = []

    dual_network = ReversiDualNetwork()

    for i in range(game_count):
        print(f'SelfMatch {i+1}/{game_count} starting...')
        r = _self_match_impl(dual_network, temperature, evaluation_count)
        record.extend(r)

        print(f'SelfMatch {i+1}/{game_count} end')

    dual_network.clear_session
    del dual_network

    return record
