from sympy import Symbol
from sympy.core.evalf import INF

from omniqubo.vars import VarAbs


class VarAbsSympyOpt(VarAbs):
    def __init__(self, name: str) -> None:
        self.var = Symbol(name)
        super().__init__(name)


class IntVar(VarAbsSympyOpt):
    """Integer variable for SympyOpt

    If lb or ub are not specified, they are set to -INF or INF respectively.
    lb must be strictly smaller than ub.

    :param name: name of the variable
    :param lb: minimal value, defaults to None
    :param ub: maximal value, defaults to None
    """

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
        """Compare with another IntVar instance

        Variables equal if they have the same name, lb and ub.
        """
        if not isinstance(o, IntVar):
            return False
        return self.name == o.name and self.lb == o.lb and self.ub == o.ub

    def __str__(self) -> str:
        return f"Integer {self.lb} <= {self.name} <= {self.ub}"


class RealVar(VarAbsSympyOpt):
    """Real variable for SympyOpt

    If lb or ub are not specified, they are set to -INF or INF respectively.
    lb must be strictly smaller than ub.

    :param name: name of the variable
    :param lb: minimal value, defaults to None
    :param ub: maximal value, defaults to None
    """

    def __init__(self, name: str, lb: float = None, ub: float = None) -> None:
        if lb is None:
            lb = -INF
        if ub is None:
            ub = INF
        assert lb < ub
        self.lb = lb
        self.ub = ub
        super().__init__(name)

    def __eq__(self, o: object) -> bool:
        """Compare with another RealVar instance

        Variables equal if they have the same name, lb and ub.
        """
        if not isinstance(o, RealVar):
            return False
        return self.name == o.name and self.lb == o.lb and self.ub == o.ub

    def __str__(self) -> str:
        return f"Real {self.lb} <= {self.name} <= {self.ub}"


class BitVar(VarAbsSympyOpt):
    """Binary variable for SympyOpt

    :param name: name of the variable
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __eq__(self, o: object) -> bool:
        """Compare with another BitVar instance

        Variables equal if they have the same name.
        """
        if not isinstance(o, BitVar):
            return False
        return self.name == o.name

    def __str__(self) -> str:
        return f"Bit {self.name}"


class SpinVar(VarAbsSympyOpt):
    """Spin variable for SympyOpt

    :param name: name of the variable
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __eq__(self, o: object) -> bool:
        """Compare with another Spin Var instance

        Variables equal if they have the same name.
        """
        if not isinstance(o, SpinVar):
            return False
        return self.name == o.name

    def __str__(self) -> str:
        return f"Spin {self.name}"
