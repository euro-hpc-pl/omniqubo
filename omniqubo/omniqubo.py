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
    """Model conversion managing class.

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
            raise ValueError(f"Unknown backend {model_backend}")
        self.logs = []  # type: List[ConverterAbs]
        self.model_logs = []  # type: List[ModelAbs]
        self.verbatim_logs = verbatim_logs

    def _convert(self, convstep: ConverterAbs):
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
            raise ValueError(f"Unknown mode {mode}")

    def make_max(self) -> ModelAbs:
        """Transform the model into maximization problem

        Change f(x) into -f(x) if the problem was a minimization problem before.

        :return: maximization problem
        """
        self._convert(MakeMax())
        return self.model

    def make_min(self) -> ModelAbs:
        """Transform the model into minimization problem

        Change f(x) into -f(x) if the problem was a maximization problem before.

        :return: minimization problem
        """
        self._convert(MakeMin())
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

        :param names: name of the remove constraints
        :param is_regexp: specifies if names should be treated as regular expression
        :param check_constraints: specifies if constraints should be check by
            interpret
        :return: updated model
        """
        self._convert(RemoveConstraint(names, is_regexp, check_constraints))
        return self.model

    def eq_to_obj(self, names: str, is_regexp: bool = True, penalty: float = None) -> ModelAbs:
        if penalty is None:
            penalty = DEFAULT_PENALTY_VALUE
        self._convert(EqToObj(names, is_regexp, penalty))
        return self.model

    def int_to_bits(
        self, names: str, mode: str, is_regexp: bool = True, trivial_conv: bool = True
    ) -> ModelAbs:
        if trivial_conv and not is_regexp:
            self._convert(TrivialIntToBit(names, is_regexp, optional=True))

        if mode == "one-hot":
            self._convert(VarOneHot(names, is_regexp))
        else:
            raise ValueError("Uknown mode {mode}")  # pragma: no cover
        return self.model

    def bit_to_spin(self, names: str, is_regexp: bool = True, reversed: bool = False) -> ModelAbs:
        self._convert(BitToSpin(names, is_regexp, reversed))
        return self.model

    def is_qubo(self) -> bool:
        return self.model.is_qubo()

    def is_hobo(self) -> bool:
        return self.model.is_hobo()

    def is_lip(self) -> bool:
        return self.model.is_lip()

    def is_qip(self) -> bool:
        return self.model.is_qip()

    def is_qcqp(self) -> bool:
        return self.model.is_qcqp()

    def is_bm(self) -> bool:
        return self.model.is_bm()

    def is_pip(self) -> bool:
        return self.model.is_pip()

    def is_ising(self, locality: int = None) -> bool:
        return self.model.is_ising(locality=locality)
