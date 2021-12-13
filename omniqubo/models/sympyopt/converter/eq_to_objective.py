import warnings

from pandas import DataFrame

from ..constraints import ConstraintEq
from ..sympyopt import SYMPYOPT_MIN_SENSE, SympyOpt
from .abs_converter import ConverterSympyOptAbs


class EqToObj(ConverterSympyOptAbs):
    def __init__(self, name: str, penalty: float) -> None:
        self.name = name
        assert penalty >= 0
        # TODO warning for 0 penalty
        self.penalty = penalty

    def interpret(self, sample: DataFrame) -> DataFrame:
        warnings.warn("EqToObj is not analysing feasibility")
        return sample

    def convert(self, model: SympyOpt) -> SympyOpt:
        assert self.name in model.constraints.keys()
        c = model.constraints[self.name]
        assert isinstance(c, ConstraintEq)
        if model.sense == SYMPYOPT_MIN_SENSE:
            model.objective += self.penalty * (c.exprleft - c.exprright) ** 2
        else:
            model.objective -= self.penalty * (c.exprleft - c.exprright) ** 2
        model.constraints.pop(self.name)
        return model
