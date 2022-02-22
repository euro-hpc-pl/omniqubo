from warnings import warn

from pandas import DataFrame

from .converter import ConverterAbs, interpret


class EqToObj(ConverterAbs):
    """Converter for shifting the equality constraint to objective function

    Converter which removes the constraint of the form f(x) = 0 and updates
    the objective function with penalty * f(x)**2, where penalty is a
    nonnegative number. When interpreting it updates the feasibility of the
    samples according to the removed constraint.

    If is_regexp is True, then all convertible equality constraints will be
    transformed.

    :param name: name of the constraint f(x) = 0
    :param is_regexp: flag deciding if name is a string or regular expression.
    :param penalty: penalty used
    """

    def __init__(self, name: str, is_regexp: bool, penalty: float) -> None:
        self.name = name
        self.is_regexp = is_regexp

        assert penalty >= 0
        if penalty == 0:
            warn(f"penalty in EqToObj for {name} is zero")
        self.penalty = penalty
        super().__init__()


@interpret.register
def interpret_eqtoobj(samples: DataFrame, converter: EqToObj) -> DataFrame:
    for verifier in converter.data["verifiers"]:
        samples["feasible"] &= verifier(samples) == 0
    return samples
