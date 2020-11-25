from typing import Callable, List
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import numpy as np
import os


class ReversiDualNetwork:

    MODEL_DIRECTORY = './model/'
    MODEL_FILE_PATH = MODEL_DIRECTORY + 'reversi.h5'

    CNN_FILTER_COUNT = 128  # 畳み込み層のカーネル数（本家は256）
    RESNET_COUNT = 16  # 残差ブロックの数（本家は19）

    INPUT_SHAPE = (8, 8, 2)  # 入力シェイプ
    OUT_POLICY_COUNT = 65  # 出力行動数(配置先(8*8)+パス(1))
    OUT_VALUE_COUNT = 1  # 出力価値数

    EPOCHS = 100

    def __init__(self):
        if self._exists_model_file():
            self._model = load_model(ReversiDualNetwork.MODEL_FILE_PATH)
        else:
            self._model = self._create_dual_network()

    def clear_session(self):
        K.clear_session()

    def predict(self, input: np.ndarray, batch_size: int = 1) -> List[np.ndarray]:
        # a, b, c = ReversiDualNetwork.INPUT_SHAPE
        # x = input.reshape((batch_size, a, b, c))
        ps, v = self._model.predict(input, batch_size=batch_size)
        return ps[0], v[0]

    def fit(self,
            input: np.ndarray,
            target: List[np.ndarray],
            batch_size: int = 128):
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


if __name__ == '__main__':
    dual_network = ReversiDualNetwork()
    a = [0] * 64
    b = [0] * 64
    input = np.array([a, b])
    p, v = dual_network.predict(np.array([a, b]), 1)
    dual_network.clear_session()
    print(p)
    print(v)
