from abc import ABC


class VarAbs(ABC):
    """Abstract class for SympyOpt variables

    All variable objects must have name and Sympy.Symbol object.

    :param name: name of the variable
    """

    def __init__(self, name: str) -> None:
        self.name = name
