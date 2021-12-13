from abc import ABC, abstractmethod

from .model import ModelAbs


class TransiplerAbs(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def convert(self, model) -> ModelAbs:
        pass

    @abstractmethod
    def can_convert(self, model) -> bool:
        pass


def transpile(model, transipler: TransiplerAbs, check=True) -> ModelAbs:
    if check:
        assert transipler.can_convert(model)
    return transipler.convert(model)
