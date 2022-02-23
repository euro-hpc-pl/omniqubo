from abc import ABC, abstractmethod


class VarAbs(ABC):
    """Abstract class for SympyOpt variables

    All variable objects must have name and Sympy.Symbol object.

    :param name: name of the variable
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def get_lb(self):
        pass

    @abstractmethod
    def get_ub(self):
        pass
