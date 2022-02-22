from copy import deepcopy
from typing import List

from pandas.core.frame import DataFrame

from omniqubo.model import ModelAbs

from .constants import DEFAULT_PENALTY_VALUE
from .converters.converter import ConverterAbs, convert, interpret
from .converters.eq_to_objective import EqToObj
from .converters.simple_manipulation import MakeMax, MakeMin, RemoveConstraint
from .converters.varreplace import BitToSpin, TrivialIntToBit, VarOneHot
from .models.sympyopt.sympyopt import SympyOpt
from .models.sympyopt.transpiler.sympyopt_to_bqm import SympyOptToDimod
from .models.sympyopt.transpiler.transpiler import transpile


class Omniqubo:
    """Model conversion managing class

    Core class of the Omniqubo package, running transpiler, conversions
    and exports of the model, and interpreting the results.

    :param model: model to be converted
    :param verbatim_logs: flag for saving models produced with each step
    :param model_backend: backend used for conversion

    """

    def __init__(self, model, verbatim_logs: bool = False, model_backend=None) -> None:
        self.orig_model = deepcopy(model)
        if model_backend is None or model_backend == "sympyopt":
            self.model = transpile(self.orig_model)  # type: ModelAbs
        else:
            raise ValueError(f"Unknown backend {model_backend}")  # pragma: no cover
        self.logs = []  # type: List[ConverterAbs]
        self.model_logs = []  # type: List[ModelAbs]
        self.verbatim_logs = verbatim_logs
        if self.verbatim_logs:
            self.model_logs.append(deepcopy(self.model))

    def convert(self, convstep: ConverterAbs):
        """Apply the conversion on the model

        Conversion step are logged, and if varbatim_logs is true, then copy of
        updated model is stored in self.model_logs.

        :param convstep: Chosen conversion method
        :return: updated model
        """
        self.logs.append(convstep)
        self.model = convert(self.model, convstep)
        if self.verbatim_logs:
            self.model_logs.append(deepcopy(self.model))
        return self.model

    def interpret(self, samples: DataFrame) -> DataFrame:
        """Interpret optimization results

        Interpret optimization result according to conversions done to the
        model. Add extra column "feasible" for pointing if the samples are
        feasible. Values for each variable should be in separate columns.
        Variables created during the conversion process will be removed, and
        only those present in the original model will be left at the end.

        .. notes:
            "feasible" set to True does not mean that the sample is feasible
            according to the original model. On the other hand False means that
            samples are not feasible.

        :param samples: samples to be interpreted
        :return: interpreted samples with "feasible" flag
        """
        samples["feasible"] = True
        for converter in reversed(self.logs):
            samples = interpret(samples, converter)
        return samples

    def to_qubo(self, penalty: float, quadratization_strength: float):
        """Transform PIP into QUBO

        In the given order: transform inequality to equality, transform
        integers with binary encodings (with trivial int to bit conversions),
        shift equality constraint to the objective function with penalty,
        apply quadratization.

        .. note::
            Not implemented equality

        :param penalty: penalty used for shifting equality.
        :param quadratization_strength: penalty used in quadratization.
        """
        raise NotImplementedError()

    def to_hobo(self, penalty: float):
        """Transform PIP into QUBO

        In the given order: transform inequality to equality, transform
        integers with binary encodings (with trivial int to bit conversions),
        shift equality constraints to the objective function with penalty.

        .. note::
            Not implemented equality

        :param penalty: penalty used for shifting equality.
        """
        raise NotImplementedError()

    def export(self, mode: str):
        """Export the model

        Export the model in a form specified by mode, for example
        BinaryQuadraticModel from dimod.

        :param mode: specifies the type of the returned model
        :raises ValueError: if unknown mode
        :return: return the transpiled model
        """
        if mode == "bqm":
            if isinstance(self.model, SympyOpt):  # HACK
                return SympyOptToDimod().transpile(self.model)
        else:
            raise ValueError(f"Unknown mode {mode}")  # pragma: no cover

    def make_max(self) -> ModelAbs:
        """Transform the model into maximization problem

        Change f(x) into -f(x) if the problem was a minimization problem before.

        :return: maximization problem
        """
        self.convert(MakeMax())
        return self.model

    def make_min(self) -> ModelAbs:
        """Transform the model into minimization problem

        Change f(x) into -f(x) if the problem was a maximization problem before.

        :return: minimization problem
        """
        self.convert(MakeMin())
        return self.model

    def rm_constraints(
        self, names: str, is_regexp: bool = True, check_constraints: bool = False
    ) -> ModelAbs:
        """Remove constraints of given name

        If is_regexp is True, then names is considered to be a regular
        expression with convention from re package. Otherwise, converter will
        look for the constraint with such name explicitly. If check_constraints
        is False, the interpret will not update "feasible" in the samples even
        if the constraint is violated.

        .. note::
            Removing constraint produced by Omniqubo may result in badly
            interpreted samples.

        :param names: names of the remove constraints
        :param is_regexp: specifies if names should be treated as regular expression
        :param check_constraints: specifies if constraints should be check by
            interpret
        :return: updated model
        """
        self.convert(RemoveConstraint(names, is_regexp, check_constraints))
        return self.model

    def eq_to_obj(self, names: str, is_regexp: bool = True, penalty: float = None) -> ModelAbs:
        """Shift equality constraints to objective function

        If is_regexp is True, then names is considered to be a regular
        expression with convention from re package. Otherwise, converter will
        look for the constraint with such name explicitly. penalty should be
        sufficiently big nonnegative number. 0 penalty is allowed, but means
        that equality constraints will be ignored (equivalent to
        rm_constraints) except feasibility will always be checked.

        :param names: names of shifted constraints
        :param is_regexp: specifies if names should be treated as regular expression
        :param penalty: specifies the penalty of the shifted constraints.
        :return: updated model
        """
        if penalty is None:
            penalty = DEFAULT_PENALTY_VALUE
        self.convert(EqToObj(names, is_regexp, penalty))
        return self.model

    def int_to_bits(
        self, names: str, mode: str, is_regexp: bool = True, trivial_conv: bool = True
    ) -> ModelAbs:
        """Convert integer variables to expression over bits

        If is_regexp is True, then names is considered to be a regular
        expression with convention from re package. Otherwise, converter will
        look for variables with such name explicitly. mode specifies the
        encoding of the integer variables, currently "one-hot" is implemented
        only. If trivial_conv is set to True, then all integers y with bound
        lb <= y <= lb+1 will be directly converted into lb + b, where b is
        binary variable, irrespective of mode chosen.

        :param names: names of converted variables
        :param mode: encoding method used
        :param is_regexp: specifies if names should be treated as regular expression
        :param trivial_conv: specify the 2-range integer variable conversion behavior
        :raises ValueError: if mode value is not known
        :return: updated model
        """
        if trivial_conv:
            self.convert(TrivialIntToBit(names, is_regexp))

        if mode == "one-hot":
            self.convert(VarOneHot(names, is_regexp))
        else:
            raise ValueError("Uknown mode {mode}")  # pragma: no cover
        return self.model

    def bit_to_spin(self, names: str, is_regexp: bool = True, reversed: bool = False) -> ModelAbs:
        """Convert bit variables to spin variables

        If is_regexp is True, then names is considered to be a regular
        expression with convention from re package. Otherwise, converter will
        look for variables with such name explicitly. if reversed is False,
        then b <- (1-s)/2 formula is used, where b is bit and s is spin.
        Otherwise, b <- (1+s)/2 is used.

        :param names: names of the binary variables
        :param is_regexp: specifies if names should be treated as regular expression
        :param reversed: spin conversion method
        :return: updated model
        """
        self.convert(BitToSpin(names, is_regexp, reversed))
        return self.model

    def is_qubo(self) -> bool:
        """Check if model is Quadratic Unconstrained Binary Optimization (QUBO)

        Model is Quadratic Unconstrained Binary Optimization if all variables
        are bits, objective function is quadratic polynomial and there are no
        constraints.

        :return: flag stating if the model is QUBO
        """
        return self.model.is_qubo()

    def is_hobo(self) -> bool:
        """Check if model is Higher Order Binary Optimization (HOBO)

        Model is Higher Order Binary Optimization if all variables are bits,
        objective function is a polynomial and there are no constraints.

        .. note::
            every pseudo-Boolean function is in fact a pseudo-Boolean
            polynomial, however here it is required that the objective
            function is written in an explicit form

        .. note:: HOBO is sometimes called Polynomial Unconstrained Binary
            Optimization

        :return: flag stating if the model is HOBO
        """
        return self.model.is_hobo()

    def is_ilp(self) -> bool:
        """Check if model is Integer Linear Program (ILP)

        Model is ILP if all variables are bits or integers, and objective and
        constraints are linear.

        :return: flag stating if the model is ILP
        """
        return self.model.is_ilp()

    def is_qip(self) -> bool:
        """Check if model is Quadratic Integer Program (QIP)

        Model is QIP if all variables are bits or integers, objective is
        quadratic polynomial, and constraints are linear.

        :return: flag stating if the model is QIP
        """
        return self.model.is_qip()

    def is_qcqp(self) -> bool:
        """Check if model is Quadratically Constrained Quadratic Program (QCQP)

        Model is QCQP if all variables are bits or integers, objective and
        constraints are quadratic polynomials.

        :return: flag stating if the model is QCQP
        """
        return self.model.is_qcqp()

    def is_bm(self) -> bool:
        """Check if model is Binary Model (BM)

        Model is BM if all variables are bits or spins.

        :return: flag stating if the model is BM
        """
        return self.model.is_bm()

    def is_pip(self) -> bool:
        """Check if model is Polynomial Integer Program (PIP)

        Model is PIP if all variables are bits or integers, objective and
        constraints are polynomials.

        :return: flag stating if the model is PIP
        """
        return self.model.is_pip()

    def is_ising(self, locality: int = 2) -> bool:
        """Check if model is an Ising Model

        Model is Ising model if all variables are spins, objective function is
        a polynomial of at most locality order and there are no constraints.

        .. note::
            It is required that the objective function is written as a
            polynomial in an explicit form

        :param locality: maximal locality, defaults to 2
        :return: flag stating if the model is Ising model with given locality
        """
        return self.model.is_ising(locality=locality)
