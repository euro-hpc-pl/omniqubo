from abc import ABC, abstractmethod

from ..sympyopt import SympyOpt


class ConvertToSymoptAbs(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def convert(self, model) -> SympyOpt:
        pass

    @abstractmethod
    def can_convert(self, model) -> bool:
        pass
