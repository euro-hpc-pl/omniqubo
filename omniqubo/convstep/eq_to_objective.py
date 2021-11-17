from omniqubo.sympyopt import SYMPYOPT_MIN_SENSE
from omniqubo.sympyopt.constraints import ConstraintEq
from omniqubo.sympyopt.sympyopt import SympyOpt

from .stepconvclass import StepConvAbs


class EqToObj(StepConvAbs):
    def __init__(self, name: str, penalty: float) -> None:
        self.name = name
        assert penalty >= 0
        # TODO warning for 0 penalty
        self.penalty = penalty

    def interpret(self, sample):
        raise NotImplementedError

    def convert(self, model: SympyOpt):
        assert self.name in model.constraints.keys()
        c = model.constraints[self.name]
        assert isinstance(c, ConstraintEq)
        if model.sense == SYMPYOPT_MIN_SENSE:
            model.objective += self.penalty * (c.exprleft - c.exprright) ** 2
        else:
            model.objective -= self.penalty * (c.exprleft - c.exprright) ** 2
        model.constraints.pop(self.name)
        return model
