from pandas import DataFrame

from omniqubo.sympyopt import SYMPYOPT_MAX_SENSE, SYMPYOPT_MIN_SENSE, SympyOpt

from .stepconvclass import StepConvAbs


class MakeMin(StepConvAbs):
    def __init__(self) -> None:
        pass

    def interpret(self, sample: DataFrame) -> DataFrame:
        return sample

    def convert(self, model: SympyOpt) -> SympyOpt:
        if model.sense == SYMPYOPT_MAX_SENSE:
            model.minimize(-model.get_objective())
        return model


class MakeMax(StepConvAbs):
    def __init__(self) -> None:
        pass

    def interpret(self, sample: DataFrame) -> DataFrame:
        return sample

    def convert(self, model: SympyOpt) -> SympyOpt:
        if model.sense == SYMPYOPT_MIN_SENSE:
            model.maximize(-model.get_objective())
        return model


class RemoveConstraint(StepConvAbs):
    def __init__(self, name: str) -> None:
        self.name = name
        pass

    def interpret(self, sample: DataFrame) -> DataFrame:
        return sample

    def convert(self, model: SympyOpt) -> SympyOpt:
        model.constraints.pop(self.name)
        return model
