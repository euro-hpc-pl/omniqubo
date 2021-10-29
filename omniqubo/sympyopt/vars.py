from sympy import Symbol
from sympy.core.evalf import INF


class VarAbs:
    def __init__(self, name: str) -> None:
        self.name = name
        self.var = Symbol(name)


class IntVar(VarAbs):
    def __init__(self, name: str, lb: int = None, ub: int = None) -> None:
        if lb is None:
            lb = -INF
        if ub is None:
            ub = INF
        assert lb < ub
        self.lb = lb
        self.ub = ub
        super().__init__(name)


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


class BitVar(VarAbs):
    def __init__(self, name: str) -> None:
        super().__init__(name)


class SpinVar(VarAbs):
    def __init__(self, name: str) -> None:
        super().__init__(name)
