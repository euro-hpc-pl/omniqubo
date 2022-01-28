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
    def __init__(self, model, verbatim_logs: bool = False, backend=None) -> None:
        self.orig_model = deepcopy(model)
        if backend is None or backend == "sympyopt":
            self.model = transpile(self.orig_model)  # type: ModelAbs
        else:
            raise ValueError(f"Unknown backend {backend}")
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
        for converter in reversed(self.logs):
            samples = interpret(samples, converter)
        return samples

    def to_qubo(self):
        raise NotImplementedError()

    def to_hobo(self):
        raise NotImplementedError()

    def export(self, mode: str):
        if mode == "bqm":
            if isinstance(self.model, SympyOpt):  # HACK
                return SympyOptToDimod().transpile(self.model)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def make_max(self) -> ModelAbs:
        self._convert(MakeMax())
        return self.model

    def make_min(self) -> ModelAbs:
        self._convert(MakeMin())
        return self.model

    def rm_constraints(
        self, names: str, is_regexp: bool = True, check_constraints: bool = False
    ) -> ModelAbs:
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

    def is_pp(self) -> bool:
        return self.model.is_pp()

    def is_ising(self, locality: int = None) -> bool:
        return self.model.is_ising(locality=locality)
