from pandas import DataFrame

from .converter import ConverterAbs, interpret


class IneqToEq(ConverterAbs):
    """Converter for transforming the inequality into equality

    Converter which transforms the constraint of the form f(x) <= 0 into
    f(x) + s == 0, where s is a nonnegative slack variable. It assumes f(x)
    always outputs integer for any x. Similarly it transforms f(x) >= 0 into
    f(x) - s >= 0. When interpreting it updates the feasibility of the
    samples according to the inequality constraint.

    If is_regexp is True, then all convertible equality constraints will be
    transformed.

    :param name: name of the constraint f(x) = 0
    :param is_regexp: flag deciding if name is a string or regular expression.
    :param check_slack: flag deciding if slacks should be checked when interpreting
    """

    def __init__(self, name: str, is_regexp: bool, check_slack: bool) -> None:
        self.name = name
        self.is_regexp = is_regexp
        self.check_slack = check_slack
        super().__init__()


@interpret.register
def interpret_ineqtoeq(samples: DataFrame, converter: IneqToEq) -> DataFrame:
    if converter.check_slack:
        for verifier, slack_name in converter.data["verifiers"]:
            samples["feasible"] &= verifier(samples) == 0
            samples.pop(slack_name)
    else:
        for verifier, slack_name, ctype in converter.data["verifiers"]:
            if ctype == "leq":
                samples["feasible"] &= verifier(samples) <= 0
            elif ctype == "geq":
                samples["feasible"] &= verifier(samples) >= 0
            else:
                raise ValueError(f"Unknown ctype {ctype}")
            samples.pop(slack_name)

    return samples
