from abc import abstractmethod

from pandas.core.frame import DataFrame

from omniqubo.models.converter import ConverterAbs

from ..sympyopt import SympyOpt


class ConverterSympyOptAbs(ConverterAbs):
    """Abstract type for converters changing SympyOpt models."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def interpret(self, samples: DataFrame) -> DataFrame:
        """Transform samples based on the given converter.

        The interpret has to be design in correspondence to the convert method.

        :param sample: input samples
        :return: transformed samples
        """
        pass

    @abstractmethod
    def convert(self, model: SympyOpt) -> SympyOpt:
        """Convert the optimization model based on the converter method.

        :param model: model to be transformed
        :return: transformed model
        """
        pass
