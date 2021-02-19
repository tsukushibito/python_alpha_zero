from typing import List
from alpha_zero.reversi import ReversiState, ai
import threading
import time
import numpy as np


def test_threading():
    class Test:
        def __init__(self) -> None:
            self._condition: threading.Condition = threading.Condition()
            self._datas: List[int] = []
            self._converted_datas: List[int] = []

        def add_data(self, data: int) -> int:
            index = len(self._datas)
            self._datas.append(data)

            if len(self._datas) < 10:
                with self._condition:
                    self._condition.wait()
            else:
                self._converted_datas = list(map(lambda v: v * v, self._datas))
                with self._condition:
                    self._condition.notify_all()

            return self._converted_datas[index]

    t = Test()

    threads = []
    for _ in range(10):
        results = []
        for i in range(10):
            def f(data) -> int:
                result = t.add_data(data)
                results.append((data, result))
            th = threading.Thread(target=f, args=(i,))
            th.start()
            threads.append(th)

        for th in threads:
            th.join()

        print(results)

    b1 = np.arange(8 * 8).reshape((8, 8))
    b2 = np.arange(8 * 8).reshape((8, 8))
    a = np.array([b1, b2])
    print(a)
    a = a.reshape((2, 8, 8))
    a = a.transpose()

    print(a)
    print(np.shape(a))


def test_predictor():
    batch_size = 64
    predictor = ai.ReversiDualNetworkPredictor(batch_size)

    state = ReversiState()
    input = np.array([state.player0_board, state.player1_board]
                     ).reshape((2, 8, 8)).transpose((1, 2, 0))

    def predict():
        p, v = predictor.predict(input)
        print(p)
        print(v)

    threads = []
    begin_t = time.time()
    for _ in range(batch_size):
        thread = threading.Thread(target=predict)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    end_t = time.time()

    print('end: ' + str(end_t - begin_t) + 's')
