from typing import List
import threading


def test_threading():
    class Test:
        def __init__(self) -> None:
            self._task_condition: threading.Condition = threading.Condition()
            self._result_condition: threading.Condition = threading.Condition()
            self._datas: List[int] = []
            self._converted_datas: List[int] = []
            self._is_end: bool = False
            self._task_thread: threading.Thread

        def run(self) -> None:
            self._task_thread = threading.Thread(target=self._task)
            self._task_thread.start()

        def add_data(self, data: int) -> int:
            self._datas.append(data)
            index = len(self._datas) - 1
            if len(self._datas) >= 10:
                with self._task_condition:
                    self._task_condition.notify_all()
            with self._result_condition:
                self._result_condition.wait()
            return self._converted_datas[index]

        def end(self) -> None:
            self._is_end = True
            with self._task_condition, self._result_condition:
                self._task_condition.notify_all()
                self._result_condition.notify_all()

        def _task(self):
            with self._task_condition:
                while not self._is_end:
                    if len(self._datas) < 10:
                        self._task_condition.wait()
                    else:
                        print('converting')
                        self._converted_datas = [i**2 for i in self._datas]
                        print('converted')
                        self._datas.clear()
                        with self._result_condition:
                            self._result_condition.notify_all()

    t = Test()
    t.run()

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

    t.end()
