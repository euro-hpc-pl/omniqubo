from abc import ABC, abstractmethod

from pandas.core.frame import DataFrame

from omniqubo.sympyopt.sympyopt import SympyOpt


class StepConvAbs(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def interpret(self, sample: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def convert(self, model: SympyOpt) -> SympyOpt:
        pass
