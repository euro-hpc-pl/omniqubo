from abc import abstractmethod

from pandas.core.frame import DataFrame

from omniqubo.models.converter import ConverterAbs

from ..sympyopt import SympyOpt


class ConverterSympyOptAbs(ConverterAbs):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def interpret(self, sample: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def convert(self, model: SympyOpt) -> SympyOpt:
        pass
