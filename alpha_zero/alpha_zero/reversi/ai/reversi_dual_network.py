from typing import Callable, List, Tuple
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from ..reversi_constants import *
import numpy as np
import os
import threading


class ReversiDualNetwork:

    MODEL_DIRECTORY = './model/'
    MODEL_FILE_PATH = MODEL_DIRECTORY + 'reversi.h5'

    CNN_FILTER_COUNT = 128  # 畳み込み層のカーネル数（本家は256）
    RESNET_COUNT = 16  # 残差ブロックの数（本家は19）

    INPUT_SHAPE = (REVERSI_BOARD_SIZE, REVERSI_BOARD_SIZE, 2)  # 入力シェイプ
    OUT_POLICY_COUNT = REVERSI_BOARD_SIZE * \
        REVERSI_BOARD_SIZE + 1  # 出力行動数(配置先(8*8)+パス(1))
    OUT_VALUE_COUNT = 1  # 出力価値数

    EPOCHS = 100

    def __init__(self):
        if self._exists_model_file():
            self._model = load_model(ReversiDualNetwork.MODEL_FILE_PATH)
        else:
            self._model = self._create_dual_network()

    def clear_session(self):
        """Kerasのセッションをクリア
        """
        K.clear_session()

    def predict(self, input: np.ndarray, batch_size: int = 1) -> Tuple[np.ndarray]:
        """推論

        Args:
            input (np.ndarray): 入力データ(縦, 横, 黒白)
            batch_size (int, optional): バッチサイズ. Defaults to 1.

        Returns:
            Tuple[np.ndarray]: 出力データ(ポリシー, バリュー)
        """
        ps, v = self._model.predict(input, batch_size=batch_size)
        return ps, v

    def fit(self,
            input: np.ndarray,
            target: List[np.ndarray],
            batch_size: int = 128) -> None:
        """学習

        Args:
            input (np.ndarray): 入力データ
            target (List[np.ndarray]): 教師データ
            batch_size (int, optional): バッチサイズ. Defaults to 128.
        """
        self._model.compile(
            loss=['categorical_crossentropy'], optimizer='adam')

        # 学習率
        def step_decay(epoch):
            x = 0.001
            if epoch >= 50:
                x = 0.0005
            if epoch >= 80:
                x = 0.00025
            return x
        lr_decay = LearningRateScheduler(step_decay)

        # 出力
        print_callback = LambdaCallback(
            on_epoch_begin=lambda epoch, logs:
            print('\rTrain {}/{}'.format(epoch + 1, ReversiDualNetwork.EPOCHS), end=''))

        # 学習の実行
        self._model.fit(input, target,
                        batch_size=batch_size,
                        epochs=ReversiDualNetwork.EPOCHS,
                        verbose=0,
                        callbacks=[lr_decay, print_callback])
        print('')

        self._model.save(ReversiDualNetwork.MODEL_FILE_PATH)
        K.clear_session()

    def _exists_model_file(self) -> bool:
        os.path.exists(ReversiDualNetwork.MODEL_FILE_PATH)

    def _create_dual_network(self) -> None:
        # デュアルネットワークの作成

        # 入力層
        input = Input(shape=ReversiDualNetwork.INPUT_SHAPE)

        # 畳み込み層
        x = self._create_conv_layer(ReversiDualNetwork.CNN_FILTER_COUNT)(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 残差ブロック x 16
        for _ in range(ReversiDualNetwork.RESNET_COUNT):
            x = self._create_residual_block(x)

        # プーリング層
        x = GlobalAveragePooling2D()(x)

        # ポリシー出力
        p = Dense(ReversiDualNetwork.OUT_POLICY_COUNT,
                  kernel_regularizer=l2(0.0005),
                  activation='softmax', name='pi')(x)

        # バリュー出力
        v = Dense(ReversiDualNetwork.OUT_VALUE_COUNT,
                  kernel_regularizer=l2(0.0005))(x)
        v = Activation('tanh', name='v')(v)

        # モデルの作成
        model = Model(inputs=input, outputs=[p, v])

        # モデルの保存
        os.makedirs(ReversiDualNetwork.MODEL_DIRECTORY,
                    exist_ok=True)  # フォルダがない時は生成
        model.save(ReversiDualNetwork.MODEL_FILE_PATH)  # ベストプレイヤーのモデル

        return model

    def _create_conv_layer(self, filters) -> Conv2D:
        # 畳み込み層の作成
        return Conv2D(filters, 3, padding='same', use_bias=False,
                      kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

    def _create_residual_block(self, x: Layer) -> Layer:
        # 残差ブロックの作成
        sc = x
        x = self._create_conv_layer(ReversiDualNetwork.CNN_FILTER_COUNT)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self._create_conv_layer(ReversiDualNetwork.CNN_FILTER_COUNT)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation('relu')(x)
        return x


class ReversiDualNetworkPredictor:
    def __init__(self, batch_size: int) -> None:
        self._dual_network: ReversiDualNetwork = ReversiDualNetwork()
        self._batch_size: int = batch_size
        self._inputs: List[np.ndarray] = []
        self._condition: threading.Condition = threading.Condition()
        self._policies: np.ndarray
        self._values: np.ndarray

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, v):
        self._batch_size = v

    def predict(self, input: np.ndarray) -> Tuple[np.ndarray]:
        index = len(self._inputs)
        self._inputs.append(input)
        print('[predict()] index: ' + str(index))

        if len(self._inputs) < self._batch_size:
            with self._condition:
                print('[predict()] wait')
                self._condition.wait()
        else:
            inputs = np.array(self._inputs)
            print('[predict()] predict, shape: ' + str(np.shape(inputs)))
            self._policies, self._values = self._dual_network.predict(
                inputs, self._batch_size)
            with self._condition:
                self._condition.notify_all()

        return self._policies[index], self._values[index]


if __name__ == '__main__':
    dual_network = ReversiDualNetwork()
    a = [0] * 64
    b = [0] * 64
    input = np.array([a, b])
    p, v = dual_network.predict(np.array([a, b]), 1)
    dual_network.clear_session()
    print(p)
    print(v)
