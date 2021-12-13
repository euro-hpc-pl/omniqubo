from abc import ABC, abstractmethod

from pandas.core.frame import DataFrame

from .model import ModelAbs


class ConverterAbs(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def interpret(self, sample: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def convert(self, model) -> ModelAbs:
        pass
