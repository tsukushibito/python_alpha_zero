from typing import List
# from alpha_zero.reversi.ai import ReversiDualNetwork
import threading
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

    # network = ReversiDualNetwork()
    # input = np.arange(8 * 8 * 2).reshape(1, 8, 8, 2)
    # print(np.shape(input))
    # p, v = network.predict(input)

    # print(np.shape(p))
    # print(np.shape(v))
