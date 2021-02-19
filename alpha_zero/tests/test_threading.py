from typing import List
from alpha_zero.reversi import ReversiState, ReversiAction, ai
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
    batch_size = 32
    predictor = ai.ReversiDualNetworkPredictor(batch_size)

    threads = []
    end_flags = {}

    def play(index: int):
        end_flags[threading.get_ident()] = False
        state = ReversiState()
        if index > 0:
            state = state.apply_action(ReversiAction(5, 3))

        if index > 1:
            state = state.apply_action(ReversiAction(5, 4))

        if index > 2:
            state = state.apply_action(ReversiAction(4, 5))

        print(state.to_string())
        mcts = ai.ReversiMcts(predictor)
        while not all(end_flags.values()):
            if not state.is_end:
                policies = mcts.search(
                    state, temperature=1.0, evaluation_count=4)
                msg = list(
                    map(lambda p: str(p[0].pos) + ':' + str(p[1]), policies))
                print('[' + str(threading.get_ident()) + '] ' + str(msg))
                actions = [p[0] for p in policies]
                scores = [p[1] for p in policies]
                action = np.random.choice(actions, p=scores)
                state = state.apply_action(action)
                if state.is_end:
                    end_flags[threading.get_ident()] = True
                    print('[' + str(threading.get_ident()) + '] end')
                    print(end_flags)
            else:
                dummy = np.arange(8 * 8 * 2).reshape((8, 8, 2))
                predictor.predict(dummy)
                # print('[' + str(i) + ']\n' + state.to_string())

        predictor.cancel()

        print('[' + str(threading.get_ident()) + ']\n' + state.to_string())

    begin_t = time.time()
    for i in range(batch_size):
        thread = threading.Thread(target=play, args=[i])
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    end_t = time.time()

    print('end: ' + str(end_t - begin_t) + 's')
