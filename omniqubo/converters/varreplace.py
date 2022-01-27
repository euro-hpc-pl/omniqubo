from typing import Any, Dict

from .converter import ConverterAbs


class VarReplace(ConverterAbs):
    """Replaces the variable with some formula.

    Transform all occurrences of the variable varname in the model with the binary
    formula, and add extra constraint if required. This is an abstract class
    which can be used for various integer encodings.

    .. note::
        Variable varname should disappear from the model, including its list of
        variables.

    :param varname: variable to be replaced
    """

    def __init__(self, varname: str) -> None:
        self.varname = varname
        self.data = dict()  # type: Dict[str, Any]


class VarOneHot(VarReplace):
    """Replace integer variables with one-hot encoding.

    Replaces integer variables with one-hot encoding, and add constraint that
    sum of new added bits is equal to one. For variable lb <= y <= ub
    the encoding creates ub-lb+1 binary variables. The limits of y needs to be
    finite integer numbers.

    :param varname: the replaced integer variable
    """

    def __init__(self, varname: str) -> None:
        super().__init__(varname)


class TrivialIntToBit(VarReplace):
    """Replace integer with binary variable.

    Replaces integer variables y with binary variable lb + b, where
    lb <= y <= lb+1 is assumed. lb should be finite integer number.

    :param varname: the replaced integer variable
    """

    def __init__(self, varname: str) -> None:
        super().__init__(varname)


class BitToSpin(VarReplace):
    """Replace binary variable with spin variable.

    Replaces binary variable b with spin variable s. The formula is (1-s)/2
    if reversed is set to true, or (1+s)/2 otherwise.

    :param varname: the replaced binary variable
    :param reversed: the flag denoting which formula is used for replacement
    """

    def __init__(self, varname: str, reversed: bool) -> None:
        super().__init__(varname)
        self.reversed = reversed
