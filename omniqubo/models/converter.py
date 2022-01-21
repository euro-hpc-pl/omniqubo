from abc import ABC, abstractmethod

from pandas.core.frame import DataFrame

from .model import ModelAbs


class ConverterAbs(ABC):
    """Abstract Converter class

    Converter is modifying the ModelAbs object in order to transform it to
    desirable program, for example QUBO. The method convert is responsible for
    converting model, while interpret should transform DataFrame samples data in
    compliance with the conversion.

    If the samples will no longer be feasible, interpret should change the
    "feasible" column in the DataFrame.
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def interpret(self, sample: DataFrame) -> DataFrame:
        """Abstract method which interprets the results

        Transform DataFrame samples data in compliance with the conversion
        method. If the samples will no longer be feasible, interpret should
        change the  "feasible" column in the DataFrame.

        :param sample: optimized results
        :return: transformed results
        """
        pass

    @abstractmethod
    def convert(self, model) -> ModelAbs:
        """Abstract method which converts the optimization model.

        :param model: model to by transformed
        :return: transformed model
        """
        pass
