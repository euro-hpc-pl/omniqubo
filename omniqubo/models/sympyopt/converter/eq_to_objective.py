import warnings

from pandas import DataFrame

from ..constraints import ConstraintEq
from ..sympyopt import SYMPYOPT_MIN_SENSE, SympyOpt
from .abs_converter import ConverterSympyOptAbs


class EqToObj(ConverterSympyOptAbs):
    """Converter for shifting the equality constraint to objective function.

    Converter which removes the constraint of the form f(x) = 0 and updates
    the objective function with penalty * f(x)**2, where penalty is a
    nonnegative number. When interpreting it updates the feasibility of the
    samples according to the removed constraint.

    :param name: name of the constraint f(x) = 0
    :param penalty: penalty used
    """

    def __init__(self, name: str, penalty: float) -> None:
        self.name = name
        assert penalty >= 0
        # TODO warning for 0 penalty
        self.penalty = penalty

    def interpret(self, samples: DataFrame) -> DataFrame:
        """Interpret samples as either feasible or not.

        :param samples: optimization results
        :return: optimization result with updated `feasibility` attribute
        """
        warnings.warn("EqToObj is not analysing feasibility")
        return samples

    def convert(self, model: SympyOpt) -> SympyOpt:
        """Shift the constraint to objective function.

        :param model: model to transform
        :return: transformed model
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
