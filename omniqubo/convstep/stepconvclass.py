from abc import ABC, abstractmethod


class StepConvAbs(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def interpret(self, sample):
        pass

    @abstractmethod
    def convert(self, model):
        pass
