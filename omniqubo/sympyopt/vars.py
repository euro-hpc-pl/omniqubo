from typing import Union

from sympy import Symbol
from sympy.core.evalf import INF


class VarAbs:
    def __init__(self, name: str) -> None:
        self.name = name
        self.var = Symbol(name)


class IntVar(VarAbs):
    def __init__(self, name: str, lb: int, ub: int) -> None:
        assert lb < ub
        self.lb = lb
        self.ub = ub
        super().__init__(name)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, IntVar):
            return False
        return self.name == o.name and self.lb == o.lb and self.ub == o.ub

    def __str__(self) -> str:
        return f"Integer {self.lb} <= {self.name} <= {self.ub}"


class RealVar(VarAbs):
    def __init__(self, name: str, lb: int = None, ub: int = None) -> None:
        if lb is None:
            lb = -INF
        if ub is None:
            ub = INF
        assert lb < ub
        self.lb = lb
        self.ub = ub
        super().__init__(name)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, RealVar):
            return False
        return self.name == o.name and self.lb == o.lb and self.ub == o.ub

    def __str__(self) -> str:
        return f"Real {self.lb} <= {self.name} <= {self.ub}"


class BitVar(VarAbs):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, BitVar):
            return False
        return self.name == o.name

    def __str__(self) -> str:
        return f"Bit {self.name}"


class SpinVar(VarAbs):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SpinVar):
            return False
        return self.name == o.name

    def __str__(self) -> str:
        return f"Spin {self.name}"


ConcreteVars = Union[IntVar, BitVar, SpinVar, RealVar]
