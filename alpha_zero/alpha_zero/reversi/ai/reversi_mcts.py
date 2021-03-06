from typing import List, Tuple
from math import sqrt
from functools import reduce
from tensorflow.keras.models import load_model
from pathlib import Path
from ..reversi_state import ReversiState
from ..reversi_action import ReversiAction
from .reversi_dual_network import ReversiDualNetwork, ReversiDualNetworkPredictor
import numpy as np

DEFAULT_EVALUATION_COUNT = 50  # 1推論あたりのシミュレーション回数（本家は1600）


class _ReversiMctsNode:
    """リバーシ用MCTSノードクラス
    """

    C_PUCT = 1.0

    # ノードの初期化
    def __init__(self, state: ReversiState, policy: float, predictor: ReversiDualNetworkPredictor):
        self.state: ReversiState = state  # 状態
        self.policy: float = policy  # 方策
        self.predictor: ReversiDualNetworkPredictor = predictor
        self.cumulative_value: float = 0  # 累計価値
        self.sim_count: int = 0  # 試行回数
        self.child_nodes: List[_ReversiMctsNode] = None  # 子ノード群

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

            if self.predictor.batch_size > 1:
                # バッチ処理による並列実行に対応するため推論処理だけ行う
                input = self._create_model_input()
                self.predictor.predict(input)

            return value

        # 子ノードが存在しない時
        if not self.child_nodes:
            # ニューラルネットワークの推論で方策と価値を取得
            input = self._create_model_input()
            policies, value = self.predictor.predict(input)

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
                    _ReversiMctsNode(self.state.apply_action(action),
                                     policy,
                                     self.predictor))
            return value

        # 子ノードが存在する時
        else:
            # アーク評価値が最大の子ノードの評価で価値を取得
            value = -self._select_child_node().evaluate()

            # 累計価値と試行回数の更新
            self.cumulative_value += value
            self.sim_count += 1
            return value

    def _create_model_input(self):
        if self.state.current_player == 0:
            input = np.array(
                [self.state.player0_board, self.state.player1_board])
        else:
            input = np.array(
                [self.state.player1_board, self.state.player0_board])
        r, c, d = ReversiDualNetwork.INPUT_SHAPE
        input = input.reshape((d, r, c)).transpose((1, 2, 0))
        return input

    # アーク評価値が最大の子ノードを取得
    def _select_child_node(self):
        # アーク評価値の計算
        # 子ノードの累計試行回数
        t = reduce(lambda a, b: a + b.sim_count, self.child_nodes, 0)
        pucb_values = []
        for child_node in self.child_nodes:
            w = child_node.cumulative_value
            n = child_node.sim_count
            p = child_node.policy
            v = (-w/n if n != 0 else 0) + \
                _ReversiMctsNode.C_PUCT * p * (sqrt(t)/(1 + n))
            pucb_values.append(v)

        # アーク評価値が最大の子ノードを返す
        return self.child_nodes[np.argmax(pucb_values)]


class ReversiMcts:
    def __init__(self,
                 predictor: ReversiDualNetworkPredictor) -> None:
        self._predictor = predictor

    def search(self,
               state: ReversiState,
               temperature: float,
               evaluation_count: int = DEFAULT_EVALUATION_COUNT) -> List[Tuple[ReversiAction, float]]:
        # 現在の局面のノードの作成
        root_node = _ReversiMctsNode(state, 0, self._predictor)

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

        policies = []
        for action, score in zip(state.allowed_actions, scores):
            policies.append((action, score))

        return policies


def search_with_mtcs(dual_network: ReversiDualNetwork,
                     state: ReversiState,
                     temperature: float,
                     evaluation_count: int = DEFAULT_EVALUATION_COUNT) -> List[float]:
    # 現在の局面のノードの作成
    root_node = _ReversiMctsNode(state, 0, dual_network)

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
