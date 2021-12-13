from pandas import DataFrame

from ..sympyopt import SYMPYOPT_MAX_SENSE, SYMPYOPT_MIN_SENSE, SympyOpt
from .abs_converter import ConverterSympyOptAbs


class MakeMin(ConverterSympyOptAbs):
    def __init__(self) -> None:
        """
        MakeMin transforms the objective function f(x) into -f(x) for
        maximization problems.
        """
        pass

    def interpret(self, sample: DataFrame) -> DataFrame:
        """
        Interprets the optimization results. Does not change the sample.

        Args:
            sample (DataFrame): optimization results

        Returns:
            DataFrame: the same optimization results
        """
        return sample

    def convert(self, model: SympyOpt) -> SympyOpt:
        """
        Transforms the objective function f(x) into -f(x) for
        maximization problems.

        Args:
            model (SympyOpt): input model

        Returns:
            SympyOpt: transformed model
        """
        if model.sense == SYMPYOPT_MAX_SENSE:
            model.minimize(-model.get_objective())
        return model


class MakeMax(ConverterSympyOptAbs):
    def __init__(self) -> None:
        """
        MakeMax transforms the objective function f(x) into -f(x) for
        maximization problems.
        """
        pass

    def interpret(self, sample: DataFrame) -> DataFrame:
        """
        Interprets the optimization results. Does not change the sample.

        Args:
            sample (DataFrame): optimization results

        Returns:
            DataFrame: the same optimization results
        """
        return sample

    def convert(self, model: SympyOpt) -> SympyOpt:
        if model.sense == SYMPYOPT_MIN_SENSE:
            model.maximize(-model.get_objective())
        return model


class RemoveConstraint(ConverterSympyOptAbs):
    def __init__(self, name: str) -> None:
        """
        RemoveConstraint remove a constraint.

        Args:
            name (str): name of constraint to be removed
        """
        self.name = name
        pass

    def interpret(self, sample: DataFrame) -> DataFrame:
        """
        Interprets the optimization results. Does not change the sample.

        Args:
            sample (DataFrame): optimization results

        Returns:
            DataFrame: the same optimization results
        """
        return sample

    def convert(self, model: SympyOpt) -> SympyOpt:
        """
        Remove a constraint self.name

        Args:
            model (SympyOpt): input model

        Returns:
            SympyOpt: model without the constraint
        """
        model.constraints.pop(self.name)
        return model
