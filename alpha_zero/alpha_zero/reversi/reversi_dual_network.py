from typing import Callable, List
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import numpy as np
import os


class ReversiDualNetwork:

    MODEL_DIRECTORY = './model/'
    MODEL_FILE_PATH = MODEL_DIRECTORY + 'model.h5'

    CNN_FILTER_COUNT = 128  # 畳み込み層のカーネル数（本家は256）
    RESNET_COUNT = 16  # 残差ブロックの数（本家は19）

    INPUT_SHAPE = (8, 8, 2)  # 入力シェイプ
    OUT_POLICY_COUNT = 65  # 出力行動数(配置先(8*8)+パス(1))
    OUT_VALUE_COUNT = 1  # 出力価値数

    def __init__(self):
        if self._exists_model_file():
            self._model = load_model(ReversiDualNetwork.MODEL_FILE_PATH)
        else:
            self._model = self._create_dual_network()

    def clear_session(self):
        K.clear_session()

    def predict(self, input: np.ndarray) -> List[np.ndarray]:
        a, b, c = ReversiDualNetwork.INPUT_SHAPE
        batch_size = 1
        x = input.reshape((batch_size, a, b, c))
        return self._model.predict(x, batch_size=batch_size)

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


if __name__ == '__main__':
    dual_network = ReversiDualNetwork()
    a = [0] * 64
    b = [0] * 64
    p, v = dual_network.predict(np.array([a, b]))
    dual_network.clear_session()
    print(p)
    print(v)
