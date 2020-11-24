from typing import List
from math import sqrt
from functools import reduce
from tensorflow.keras.models import load_model
from pathlib import Path
from ..game import GameState
from .reversi_dual_network import ReversiDualNetwork
from .reversi_state import ReversiState
import numpy as np

# パラメータの準備
PV_EVALUATE_COUNT = 50  # 1推論あたりのシミュレーション回数（本家は1600）


class ReversiMctsNode:
    # モンテカルロ木探索のノードの定義
    C_PUCT = 1.0

    # ノードの初期化
    def __init__(self, state: GameState, policy: float, dual_network: ReversiDualNetwork):
        self.state: GameState = state  # 状態
        self.policy: float = policy  # 方策
        self.dual_network: ReversiDualNetwork = dual_network
        self.cumulative_value: float = 0  # 累計価値
        self.sim_count: int = 0  # 試行回数
        self.child_nodes: List[ReversiMctsNode] = None  # 子ノード群

    # 局面の価値の計算
    def evaluate(self):
        # ゲーム終了時
        if self.state.is_end():
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
            temp = np.array([self.state.player0_board,
                             self.state.player1_board])
            r, c, s = ReversiDualNetwork.INPUT_SHAPE
            input = temp.reshape((s, r, c)).transpose()
            policies, value = self.dual_network.predict(input)

            # 累計価値と試行回数の更新
            self.cumulative_value += value
            self.sim_count += 1

            # 子ノードの展開
            allowed_actions = map(
                lambda a: a.value, self.state.allowed_actions)
            policies = policies[allowed_actions]  # 合法手だけに変換
            sum_p = sum(policies)
            policies /= sum_p if sum_p != 0 else 1  # 合計が1になるように補正
            self.child_nodes = []
            for action, policy in zip(self.state.allowed_actions, policies):
                self.child_nodes.append(
                    ReversiMctsNode(self.state.take_action(action),
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
        t = reduce(lambda a, b: a.sim_count + b.sim_count, self.child_nodes)
        pucb_values = []
        for child_node in self.child_nodes:
            w = child_node.cumulative_value
            n = child_node.sim_count
            p = child_node.policy
            v = (w/n if n != 0 else 0) + \
                ReversiMctsNode.C_PUCT * p * (sqrt(t)/(1 + n))
            v *= -1
            pucb_values.append(v)

        # アーク評価値が最大の子ノードを返す
        return self.child_nodes[np.argmax(pucb_values)]


def reversi_mtcs(dual_network: ReversiDualNetwork, state: ReversiState, temperature: float):
    # 現在の局面のノードの作成
    root_node = Node(state, 0)

    # 複数回の評価の実行
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()

    # 合法手の確率分布
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:  # 最大値のみ1
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:  # ボルツマン分布でバラつき付加
        scores = boltzman(scores, temperature)
    return scores


def pv_mcts_scores(model, state, temperature):
    # モンテカルロ木探索のスコアの取得
    # モンテカルロ木探索のノードの定義
    class Node:
        # ノードの初期化
        def __init__(self, state, p):
            self.state = state  # 状態
            self.p = p  # 方策
            self.w = 0  # 累計価値
            self.n = 0  # 試行回数
            self.child_nodes = None  # 子ノード群

        # 局面の価値の計算
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                # 勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # ニューラルネットワークの推論で方策と価値を取得
                policies, value = predict(model, self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(
                        Node(self.state.next(action), policy))
                return value

            # 子ノードが存在する時
            else:
                # アーク評価値が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        # アーク評価値が最大の子ノードを取得
        def next_child_node(self):
            # アーク評価値の計算
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                                   C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))

            # アーク評価値が最大の子ノードを返す
            return self.child_nodes[np.argmax(pucb_values)]

    # 現在の局面のノードの作成
    root_node = Node(state, 0)

    # 複数回の評価の実行
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()

    # 合法手の確率分布
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:  # 最大値のみ1
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:  # ボルツマン分布でバラつき付加
        scores = boltzman(scores, temperature)
    return scores

# モンテカルロ木探索で行動選択


def pv_mcts_action(model, temperature=0):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

# ボルツマン分布


def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]


# 動作確認
if __name__ == '__main__':
    # モデルの読み込み
    path = sorted(Path('./model').glob('*.h5'))[-1]
    model = load_model(str(path))

    # 状態の生成
    state = State()

    # モンテカルロ木探索で行動取得を行う関数の生成
    next_action = pv_mcts_action(model, 1.0)

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 行動の取得
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

        # 文字列表示
        print(state)


class ReversiMcts:
    pass
