from typing import List
from math import sqrt
from functools import reduce
from tensorflow.keras.models import load_model
from pathlib import Path
from ..reversi_state import ReversiState
from ..reversi_action import ReversiAction
from .reversi_dual_network import ReversiDualNetwork
import numpy as np

DEFAULT_EVALUATION_COUNT = 50  # 1推論あたりのシミュレーション回数（本家は1600）


class ReversiMctsNode:
    # モンテカルロ木探索のノードの定義
    C_PUCT = 1.0

    # ノードの初期化
    def __init__(self, state: ReversiState, policy: float, dual_network: ReversiDualNetwork):
        self.state: ReversiState = state  # 状態
        self.policy: float = policy  # 方策
        self.dual_network: ReversiDualNetwork = dual_network
        self.cumulative_value: float = 0  # 累計価値
        self.sim_count: int = 0  # 試行回数
        self.child_nodes: List[ReversiMctsNode] = None  # 子ノード群

    # 局面の価値の計算
    def evaluate(self):
        # ゲーム終了時
        if self.state.is_end:
            # 勝敗結果で価値を取得
            value = 0
            if self.state.is_current_player_winner:
                value = 1
            elif self.state.is_draw:
                value = 0
            else:
                value = -1

            # 累計価値と試行回数の更新
            self.cumulative_value += value
            self.sim_count += 1
            return value

        # 子ノードが存在しない時
        if not self.child_nodes:
            # ニューラルネットワークの推論で方策と価値を取得
            policies, value = self.dual_network.predict(
                np.array([self.state.to_model_input()]), 1)

            # 累計価値と試行回数の更新
            self.cumulative_value += value
            self.sim_count += 1

            # 子ノードの展開
            allowed_actions = [a.value for a in self.state.allowed_actions]
            policies = policies[allowed_actions]  # 合法手だけに変換
            sum_p = sum(policies)
            policies /= sum_p if sum_p != 0 else 1  # 合計が1になるように補正
            self.child_nodes = []
            for action, policy in zip(self.state.allowed_actions, policies):
                self.child_nodes.append(
                    ReversiMctsNode(self.state.apply_action(action),
                                    policy,
                                    self.dual_network))
            return value

        # 子ノードが存在する時
        else:
            # アーク評価値が最大の子ノードの評価で価値を取得
            value = -self.select_child_node().evaluate()

            # 累計価値と試行回数の更新
            self.cumulative_value += value
            self.sim_count += 1
            return value

    # アーク評価値が最大の子ノードを取得
    def select_child_node(self):
        # アーク評価値の計算
        # 子ノードの累計試行回数
        t = reduce(lambda a, b: a + b.sim_count, self.child_nodes, 0)
        pucb_values = []
        for child_node in self.child_nodes:
            w = child_node.cumulative_value
            n = child_node.sim_count
            p = child_node.policy
            v = (-w/n if n != 0 else 0) + \
                ReversiMctsNode.C_PUCT * p * (sqrt(t)/(1 + n))
            pucb_values.append(v)

        # アーク評価値が最大の子ノードを返す
        return self.child_nodes[np.argmax(pucb_values)]


def search_with_mtcs(dual_network: ReversiDualNetwork,
                     state: ReversiState,
                     temperature: float,
                     evaluation_count: int = DEFAULT_EVALUATION_COUNT) -> List[float]:
    # 現在の局面のノードの作成
    root_node = ReversiMctsNode(state, 0, dual_network)

    # 複数回の評価の実行
    for _ in range(evaluation_count):
        root_node.evaluate()

    # 合法手の確率分布(試行回数が多い手を採用)
    scores = [n.sim_count for n in root_node.child_nodes]
    if temperature == 0:  # 最大値のみ1
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        # ボルツマン分布でバラつき付加
        def boltzman(xs, temperature):
            xs = [x ** (1 / temperature) for x in xs]
            return [x / sum(xs) for x in xs]

        scores = boltzman(scores, temperature)

    return scores


def choice_next_action(allowed_actions: List[ReversiAction], scores: List[float]) -> ReversiAction:
    return np.random.choice(allowed_actions, p=scores)


# 動作確認
if __name__ == '__main__':
    dual_network = ReversiDualNetwork()

    # 状態の生成
    state = ReversiState()

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_end:
            break

        # 行動の取得
        action = search_with_mtcs(dual_network, state, 1.0)

        # 次の状態の取得
        state = state.apply_action(action)

        # 文字列表示
        print(state)
