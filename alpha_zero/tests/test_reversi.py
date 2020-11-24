﻿from alpha_zero.game import Game
from alpha_zero.reversi import ReversiDualNetwork
from alpha_zero.reversi import ReversiState
from alpha_zero.reversi import search_with_mtcs
from alpha_zero.reversi import choice_next_action
from alpha_zero.reversi import run_self_match
from alpha_zero.reversi import save_self_match_record
from alpha_zero.reversi import load_self_match_record

import numpy as np


def test_reversi_mtcs():
    dual_network = ReversiDualNetwork()

    # 状態の生成
    state = ReversiState()

    # ゲーム終了までループ
    while not state.is_end:

        # 行動の取得
        scores = search_with_mtcs(dual_network, state, 1.0, 10)
        action = choice_next_action(state.allowed_actions, scores)

        # 次の状態の取得
        state = state.take_action(action)

        # 文字列表示
        print(state.to_string())


def test_reversi_self_play():
    record = run_self_match(1, 1.0)
    save_self_match_record(record)


if __name__ == "__main__":
    # test_reversi_mtcs()
    test_reversi_self_play()
