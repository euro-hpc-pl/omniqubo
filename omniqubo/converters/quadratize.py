from pandas import DataFrame

from .converter import ConverterAbs, interpret


class QuadratizePyqubo(ConverterAbs):
    """Quadratize a HOBO using pyqubo quadratization

    Quadratize HOBO into a QUBO respectively.

    :param strength: the strength of the reduction constraint
    """

    def __init__(self, strength: float) -> None:
        assert strength >= 0  # should be nonnegative
        self.strength = strength
        super().__init__()


@interpret.register
def interpret_quadratizepyqubo(samples: DataFrame, converter: QuadratizePyqubo) -> DataFrame:
    raise NotImplementedError()
