from pandas import DataFrame

from .converter import ConverterAbs, interpret


class MakeMin(ConverterAbs):
    """Converter for making the optimization model a minimization problem

    If the optimization model is a minimization problem, does not do
    anything, otherwise changes objective function f(x) into -f(x).
    """

    def __init__(self) -> None:
        pass


@interpret.register
def interpret_makemin(samples: DataFrame, converter: MakeMin) -> DataFrame:
    return samples


class MakeMax(ConverterAbs):
    """Converter for making the optimization model a maximization problem

    If the optimization model is a maximization problem, does not do
    anything, otherwise changes objective function f(x) into -f(x).
    """

    def __init__(self) -> None:
        pass


@interpret.register
def interpret_makemax(samples: DataFrame, converter: MakeMax) -> DataFrame:
    return samples


class RemoveConstraint(ConverterAbs):
    """Removes the given constraint

    Removes the constraint of given name if exists. Otherwise do not do
    anything to the model. If check_constraint is set to False, the interpreted
    samples are not checked against the removed constraint. If is_regexp is set
    to True, then all constraint with matching names are removed.


    :param name: the name of the removed model
    :param is_regexp: flag deciding if name is regular expression
    :param check_constraint: flag for checking the constraint
    """

    def __init__(self, name: str, is_regexp: bool, check_constraint: bool) -> None:
        self.name = name
        self.check_constraint = check_constraint
        self.is_regexp = is_regexp
        super().__init__()


@interpret.register
def interpret_removeconstraint(samples: DataFrame, converter: RemoveConstraint) -> DataFrame:
    for verifier, ctype in converter.data["verifiers"]:
        if ctype == "eq":
            samples["feasible"] &= verifier(samples) == 0
        elif ctype == "geq":
            samples["feasible"] &= verifier(samples) >= 0
        elif ctype == "leq":
            samples["feasible"] &= verifier(samples) <= 0
        else:
            ValueError(f"Uknonwn Constraint type {ctype}")  # pragma: no cover
    return samples


class SetIntVarBounds(ConverterAbs):
    """Set bounds for the variable if it is unbounded

    If lb or ub is None, then the bound is not changed. If is_regexp is set
    to True, then for all variables with matching names bounds will be
    updated. lb and ub cannot be None simultaneously.

    :param name: the name of the removed model
    :param is_regexp: flag deciding if name is regular expression
    :param lb: the lower bound, defaults to None
    :param ub: the upper bound, defaults to None
    """

    def __init__(self, name: str, is_regexp: bool, lb: int, ub: int) -> None:
        assert lb is not None or ub is not None
        if lb is not None and ub is not None:
            assert lb < ub
        self.varname = name
        self.is_regexp = is_regexp
        self.lb = lb
        self.ub = ub
        super().__init__()


@interpret.register
def interpret_setintvarbounds(samples: DataFrame, converter: SetIntVarBounds) -> DataFrame:
    return samples


class RemoveTrivialConstraints(ConverterAbs):
    """Remove trivial constraints

    Remove trivial inequalities of the form P(x) <= 0, where max P(x) <= 0 can
    be shown. Similar for other types of inequalities.

    :param name: the name of the removed model
    :param is_regexp: flag deciding if name is regular expression
    """

    def __init__(self, name: str, is_regexp: bool) -> None:
        self.name = name
        self.is_regexp = is_regexp
        super().__init__()


@interpret.register
def interpret_removetrivialconstraints(
    samples: DataFrame, converter: RemoveTrivialConstraints
) -> DataFrame:
    return samples
