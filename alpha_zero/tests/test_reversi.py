from alpha_zero.game import Game
from alpha_zero.reversi.ai import ReversiDualNetwork
from alpha_zero.reversi.ai import search_with_mtcs
from alpha_zero.reversi.ai import choice_next_action
from alpha_zero.reversi import ReversiState
from alpha_zero.reversi import run_self_match
from alpha_zero.reversi import save_self_match_record
from alpha_zero.reversi import load_last_self_match_record
from alpha_zero.reversi import record_to_model_fitting_data

import numpy as np


def test_reversi_mtcs():
    dual_network = ReversiDualNetwork()

    # 状態の生成
    state = ReversiState()

    # ゲーム終了までループ
    while not state.is_end:

        # 行動の取得
        scores = search_with_mtcs(dual_network, state, 1.0, 5)
        action = choice_next_action(state.allowed_actions, scores)

        # 次の状態の取得
        state = state.apply_action(action)

        # 文字列表示
        print(state.to_string())


def test_reversi_self_match():
    record = run_self_match(1, 1.0, 5)
    save_self_match_record(record)


def test_train_network():
    dual_network = ReversiDualNetwork()
    record = load_last_self_match_record()
    input, target = record_to_model_fitting_data(record)
    dual_network.fit(input, target)


if __name__ == "__main__":
    test_reversi_mtcs()
    # test_reversi_self_match()
    # test_train_network()
