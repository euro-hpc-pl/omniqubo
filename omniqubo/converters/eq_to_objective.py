from .converter import ConverterAbs


class EqToObj(ConverterAbs):
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
