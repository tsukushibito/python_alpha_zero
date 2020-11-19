from abc import ABCMeta, abstractclassmethod


class Game(metaclass=ABCMeta):
    def __init__(self):
        self.x = 0

    @abstractclassmethod
    def step(self, action):
        pass


class Action(metaclass=ABCMeta):
    pass
