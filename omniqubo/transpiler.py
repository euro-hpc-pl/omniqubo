from abc import ABC, abstractmethod

from .model import ModelAbs


class TransiplerAbs(ABC):
    """Abstract Transpiler class

    Transpilers are transforming the optimization models written in different
    modeling languages into the one used in Omniqubo, or vice-versa.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def transpile(self, model) -> ModelAbs:
        """Transpile model into model used in Omniqubo

        :param model: model to be transpiled
        :return: transpiled model
        """
        pass

    @abstractmethod
    def can_transpile(self, model) -> bool:
        """Check if model can be transpiled

        :type model: model to be transpiled
        :return: flag denoting if model can be transpiled
        """
        pass
