from pandas import DataFrame

from ..sympyopt import SYMPYOPT_MAX_SENSE, SYMPYOPT_MIN_SENSE, SympyOpt
from .abs_converter import ConverterSympyOptAbs


class MakeMin(ConverterSympyOptAbs):
    """Converter for making the optimization model a minimization problem.

    If the optimization model is a minimization problem, do not do
    anything, otherwise changes objective function f(x) into -f(x).
    """

    def __init__(self) -> None:
        pass

    def interpret(self, samples: DataFrame) -> DataFrame:
        """Return the same samples.

        :param samples: optimized samples
        :return: the same samples
        """
        return samples

    def convert(self, model: SympyOpt) -> SympyOpt:
        """Make the optimization model a minimization problem.

        :param model: model to be transformed
        :return: minimization model
        """
        if model.sense == SYMPYOPT_MAX_SENSE:
            model.minimize(-model.get_objective())
        return model


class MakeMax(ConverterSympyOptAbs):
    """Converter for making the optimization model a maximization problem.

    If the optimization model is a maximization problem, do not do
    anything, otherwise changes objective function f(x) into -f(x).
    """

    def __init__(self) -> None:
        pass

    def interpret(self, samples: DataFrame) -> DataFrame:
        """Return the same samples.

        :param samples: optimized samples
        :return: the same samples
        """
        return samples

    def convert(self, model: SympyOpt) -> SympyOpt:
        """Make the optimization model a maximization problem.

        :param model: model to be transformed
        :return: maximization model
        """
        if model.sense == SYMPYOPT_MIN_SENSE:
            model.maximize(-model.get_objective())
        return model


class RemoveConstraint(ConverterSympyOptAbs):
    """Removes the given constraint.

    Removes the constraint of given name if exists. Otherwise do not do
    anything to the model.

    :param name: the name of the removed model
    """

    def __init__(self, name: str) -> None:
        self.name = name
        pass

    def interpret(self, samples: DataFrame) -> DataFrame:
        """Return the same samples.

        :param samples: optimized samples
        :return: the same samples
        """
        return samples

    def convert(self, model: SympyOpt) -> SympyOpt:
        """Removes the constraint with the given name.

        :param model: model to be transformed
        :return: model with out the constraint
        """
        model.constraints.pop(self.name)
        return model
