import numpy as np
from pandas import DataFrame

from .converter import ConverterAbs, interpret
from .utils import INTER_STR_SEP


class VarReplace(ConverterAbs):
    """Replaces the variable with some formula

    Transform all occurrences of the variable varname in the model with the
    binary formula, and add extra constraint if required. This is an abstract
    class which can be used for various integer encodings. If is_regexp is set
    to True, then all appropriate variables should be replaced.

    .. note::
        Variables varname disappear from the model, including its list of
        variables.

    :param varname: variable to be replaced
    :param is_regexp: flag deciding if varname is regular expression
    """

    def __init__(self, varname: str, is_regexp: bool) -> None:
        self.varname = varname
        self.is_regexp = is_regexp
        super().__init__()


class VarOneHot(VarReplace):
    """Replace integer variables with one-hot encoding

    Replaces integer variables with one-hot encoding, and add constraint that
    sum of new added bits is equal to one. For variable lb <= y <= ub
    the encoding creates ub-lb+1 binary variables. The limits of y needs to be
    finite integer numbers. If is_regexp is set to True, then all bounded
    integer variables are replaced.

    :param varname: the replaced integer variable
    :param is_regexp: flag deciding if varname is regular expression
    """

    def __init__(self, varname: str, is_regexp: bool) -> None:
        super().__init__(varname, is_regexp)


@interpret.register
def interpret_varonehot(samples: DataFrame, converter: VarOneHot) -> DataFrame:
    for name in converter.data["bounds"].keys():
        lb, ub = converter.data["bounds"][name]

        names = [f"{name}{INTER_STR_SEP}OH_{i}" for i in range(ub - lb + 1)]
        samples["feasible_tmp"] = samples.apply(lambda row: sum(row[n] for n in names) == 1, axis=1)

        def set_var_value(row):
            return lb + [row[n] for n in names].index(1) if row["feasible_tmp"] else np.nan

        samples[name] = samples.apply(
            set_var_value,
            axis=1,
        )

        samples["feasible"] &= samples.pop("feasible_tmp")
        for n in names:
            samples.pop(n)

    return samples


class VarBinary(VarReplace):
    """Replace integer variables with binary encoding

    Replaces integer variables with binary encoding. For variable lb <= y <= ub
    the encoding creates approximately log(ub-lb+1) binary variables. The limits
    of y needs to be finite integer numbers. If is_regexp is set to True, then
    all bounded integer variables are replaced.

    :param varname: the replaced integer variable
    :param is_regexp: flag deciding if varname is a regular expression
    """

    def __init__(self, varname: str, is_regexp: bool) -> None:
        super().__init__(varname, is_regexp)


def _binary_encoding_coeff(lb: int, ub: int):
    span_size = ub - lb + 1
    is_power_of_two = span_size and (not (span_size & (span_size - 1)))
    if is_power_of_two:
        bit_no = span_size.bit_length() - 1
        vals = [2 ** i for i in range(bit_no)]
    else:
        bit_no = span_size.bit_length()
        vals = [2 ** i for i in range(bit_no - 1)]
        vals.append(ub - lb - sum(vals))
    return vals


@interpret.register
def interpret_binary(samples: DataFrame, converter: VarBinary) -> DataFrame:
    for name in converter.data["bounds"].keys():
        lb, ub = converter.data["bounds"][name]
        vals = _binary_encoding_coeff(lb, ub)
        bnames = [f"{name}{INTER_STR_SEP}BIN_{i}" for i in range(len(vals))]

        samples[name] = lb + sum(val * samples[bname] for val, bname in zip(vals, bnames))
        for n in bnames:
            samples.pop(n)
    return samples


class VarPracticalBinary(VarReplace):
    """Replace integer variables with practical binary encoding

    TODO: fill

    :param varname: the replaced integer variable
    :param is_regexp: flag deciding if varname is a regular expression
    :param ub: allowed upper bound
    """

    def __init__(self, varname: str, is_regexp: bool, ub: int) -> None:
        assert ub > 1
        super().__init__(varname, is_regexp)


@interpret.register
def interpret_varpracticalbinary(samples: DataFrame, converter: VarPracticalBinary) -> DataFrame:
    raise NotImplementedError()


class TrivialIntToBit(VarReplace):
    """Replace integer with binary variable

    Replaces integer variables y with binary variable lb + b, where
    lb <= y <= lb+1 is assumed. lb should be finite integer number. If is_regexp
    is set to True, then all integer variables satisfying the constraint above
    are replaced.

    :param varname: the replaced integer variable
    :param is_regexp: flag deciding if varname is regular expression
    :param optional: if set to True, the converts only integer variables with
        appropriate bounds
    """

    def __init__(self, varname: str, is_regexp: bool) -> None:
        super().__init__(varname, is_regexp)


@interpret.register
def interpret_trivialinttobit(samples: DataFrame, converter: TrivialIntToBit) -> DataFrame:
    for name, lb in converter.data["lb"].items():
        name_new = f"{name}{INTER_STR_SEP}itb"
        samples.rename(columns={name_new: name})
        samples[name] = samples.pop(name_new) + lb
    return samples


class BitToSpin(VarReplace):
    """Replace binary variable with spin variable

    Replaces binary variable b with spin variable s. The formula is (1-s)/2
    if reversed is set to False, or (1+s)/2 otherwise. If is_regexp is set to
    True, then all binary variables are replaced.

    :param varname: the replaced binary variable
    :param is_regexp: flag deciding if varname is regular expression
    :param reversed: the flag denoting which formula is used for replacement
    """

    def __init__(self, varname: str, is_regexp: bool, reversed: bool) -> None:
        super().__init__(varname, is_regexp)
        self.reversed = reversed


@interpret.register
def interpret_bittospin(samples: DataFrame, converter: BitToSpin) -> DataFrame:
    for name in converter.data["varnames"]:
        name_new = f"{name}{INTER_STR_SEP}bts"
        samples.rename(columns={name_new: name})
        if converter.reversed:
            samples[name] = (1 - samples.pop(name_new)) / 2
        else:
            samples[name] = (1 + samples.pop(name_new)) / 2
    return samples


class SpinToBit(VarReplace):
    """Replace spin variable with bit variable

    Replaces spin variable s with bit variable b. The formula is 1-2*b
    if reversed is set to False, or 2*b-1 otherwise. If is_regexp is set to
    True, then all binary variables are replaced.

    :param varname: the replaced spin variable
    :param is_regexp: flag deciding if varname is regular expression
    :param reversed: the flag denoting which formula is used for replacement
    """

    def __init__(self, varname: str, is_regexp: bool, reversed: bool) -> None:
        super().__init__(varname, is_regexp)
        self.reversed = reversed


@interpret.register
def interpret_spintobit(samples: DataFrame, converter: SpinToBit) -> DataFrame:
    for name in converter.data["varnames"]:
        name_new = f"{name}{INTER_STR_SEP}stb"
        samples.rename(columns={name_new: name})
        if converter.reversed:
            samples[name] = 1 - 2 * samples.pop(name_new)
        else:
            samples[name] = 1 + 2 * samples.pop(name_new)
    return samples
