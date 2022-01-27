from warnings import warn

from pandas import DataFrame

from .converter import ConverterAbs, interpret


class MakeMin(ConverterAbs):
    """Converter for making the optimization model a minimization problem.

    If the optimization model is a minimization problem, does not do
    anything, otherwise changes objective function f(x) into -f(x).
    """

    def __init__(self) -> None:
        pass


@interpret.register
def interpret_makemin(samples: DataFrame, converter: MakeMin) -> DataFrame:
    return samples


class MakeMax(ConverterAbs):
    """Converter for making the optimization model a maximization problem.

    If the optimization model is a maximization problem, does not do
    anything, otherwise changes objective function f(x) into -f(x).
    """

    def __init__(self) -> None:
        pass


@interpret.register
def interpret_makemax(samples: DataFrame, converter: MakeMax) -> DataFrame:
    return samples


class RemoveConstraint(ConverterAbs):
    """Removes the given constraint.

    Removes the constraint of given name if exists. Otherwise do not do
    anything to the model.

    :param name: the name of the removed model
    """

    def __init__(self, name: str) -> None:
        self.name = name
        pass


@interpret.register
def interpret_removeconstraint(samples: DataFrame, converter: RemoveConstraint) -> DataFrame:
    warn("Feasibility is not checked yet for RemoveConstraint")
    # TODO: add flag to RemoveConstraint
    return samples
