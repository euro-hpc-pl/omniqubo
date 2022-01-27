from abc import ABC, abstractmethod
from typing import Any, Dict
from warnings import warn

from multimethod import multimethod
from pandas import DataFrame

from omniqubo.model import ModelAbs


class ConverterAbs(ABC):
    """Abstract Converter class

    Converter is modification description of the ModelAbs object. It has not
    additional methods, however in order to allow it to be used simultaneously,
    one has to dispatched convert(model, converter) -> model and
    interpret(samples, converter) -> with each new converter or model.
    The method convert is responsible for converting model, while interpret
    transforms DataFrame samples data in compliance with the conversion.

    Optionally, one can implement can_checkc(model, converter) -> bool which
    verifies if the model can be transformed in compliance with the converter.

    If the samples will no longer be feasible, interpret should change the
    "feasible" column in the DataFrame.

    Converter with can have additional members specifying the conversion
    details.
    """

    @abstractmethod
    def __init__(self) -> None:
        self.data = {}  # type: Dict[str, Any]
        pass


@multimethod
def convert(model: ModelAbs, converter: ConverterAbs):
    raise NotImplementedError(f"{type(converter)} cannot be applied on {type(model)}")


@multimethod
def can_convert(samples: ModelAbs, converter: ConverterAbs) -> bool:
    warn(f"can_check not implemented for {type(converter)}, defaults to True", Warning)
    return True


@multimethod
def interpret(samples: DataFrame, converter: ConverterAbs) -> DataFrame:
    raise NotImplementedError(f"interpret not implemented for {type(converter)}")