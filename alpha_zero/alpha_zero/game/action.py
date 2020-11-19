from abc import ABC, abstractmethod
from typing import Any


class Action(ABC):
    @property
    @abstractmethod
    def value(self) -> int:
        pass
