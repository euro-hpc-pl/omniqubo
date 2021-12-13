from abc import abstractmethod

from pandas.core.frame import DataFrame

from omniqubo.models.converter import ConverterAbs

from ..sympyopt import SympyOpt


class ConverterSympyOptAbs(ConverterAbs):
    """
    Abstract type for converters changing SympyOpt models
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Abstract method. Should initialize the class
        """
        pass

    @abstractmethod
    def interpret(self, sample: DataFrame) -> DataFrame:
        """
        Abstract method. Should interpret the samples and
        update them based on the form of transformation.

        Args:
            sample (DataFrame): original samples

        Returns:
            DataFrame: transformed samples
        """
        pass

    @abstractmethod
    def convert(self, model: SympyOpt) -> SympyOpt:
        """
        Abstract method. Should change the model of type SympyOpt based on the
        form of transformation.

        Args:
            model (SympyOpt): original model

        Returns:
            SympyOpt: transformed model
        """
        pass
