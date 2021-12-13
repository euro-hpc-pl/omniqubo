import warnings

from pandas import DataFrame

from ..constraints import ConstraintEq
from ..sympyopt import SYMPYOPT_MIN_SENSE, SympyOpt
from .abs_converter import ConverterSympyOptAbs


class EqToObj(ConverterSympyOptAbs):
    def __init__(self, name: str, penalty: float) -> None:
        """
        EqToObj transforms SympyOpt by removing the equality constraint f(x) = 0
        and adding (penalty * f(x)**2) to objective function. Penalty should be
        nonnegative.

        Args:
            name (str): name of the manipulated constraint
            penalty (float): nonnegative value scaling the constraint
        """
        self.name = name
        assert penalty >= 0
        # TODO warning for 0 penalty
        self.penalty = penalty

    def interpret(self, sample: DataFrame) -> DataFrame:
        """
        Interprets the optimization results. Sets feasibility in sample to false
        if variables do not satisfy them

        Args:
            sample (DataFrame): optimization results

        Returns:
            DataFrame: updated optimization results
        """
        warnings.warn("EqToObj is not analysing feasibility")
        return sample

    def convert(self, model: SympyOpt) -> SympyOpt:
        """
        Transforms SympyOpt by removing the equality constraint f(x) = 0
        and adding (penalty * f(x)**2) to objective function.

        Args:
            model (SympyOpt): input model

        Returns:
            SympyOpt: transformed model
        """
        assert self.name in model.constraints.keys()
        c = model.constraints[self.name]
        assert isinstance(c, ConstraintEq)
        if model.sense == SYMPYOPT_MIN_SENSE:
            model.objective += self.penalty * (c.exprleft - c.exprright) ** 2
        else:
            model.objective -= self.penalty * (c.exprleft - c.exprright) ** 2
        model.constraints.pop(self.name)
        return model
