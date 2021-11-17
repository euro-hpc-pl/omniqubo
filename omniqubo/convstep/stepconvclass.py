from abc import ABC, abstractmethod

from omniqubo.sympyopt.sympyopt import SympyOpt


class StepConvAbs(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def interpret(self, sample):
        pass

    @abstractmethod
    def convert(self, model: SympyOpt):
        pass
