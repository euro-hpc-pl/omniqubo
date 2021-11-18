import re
from copy import deepcopy
from typing import List

from omniqubo.sympyopt.constraints import ConstraintEq

from ..convstep import EqToObj, MakeMax, MakeMin, RemoveConstraint, StepConvAbs, VarOneHot
from ..soptconv import convert_to_sympyopt
from ..sympyopt import SympyOpt
from ..sympyopt.vars import IntVar

DEFAULT_PENALTY_VALUE = 1000.0


class Omniqubo:
    def __init__(self, model, verbatim_logs: bool = False) -> None:
        self.orig_model = deepcopy(model)
        self.model = convert_to_sympyopt(self.orig_model)  # type: SympyOpt
        self.logs = []  # type: List[StepConvAbs]
        self.model_logs = []  # type: List[SympyOpt]
        self.verbatim_logs = verbatim_logs

    def _convert(self, convstep: StepConvAbs):
        self.logs.append(convstep)
        self.model = convstep.convert(self.model)
        if self.verbatim_logs:
            self.model_logs.append(deepcopy(self.model))
        return self.model

    def interpret(self, samples, general_form: bool = True):
        raise NotImplementedError()

    def to_qubo(self):
        raise NotImplementedError()

    def to_hobo(self):
        raise NotImplementedError()

    def export(self, mode: str):
        raise NotImplementedError()

    def make_max(self) -> SympyOpt:
        self._convert(MakeMax())
        return self.model

    def make_min(self) -> SympyOpt:
        self._convert(MakeMin())
        return self.model

    def rm_constraint(self, name: str = None, regname: str = None) -> SympyOpt:
        assert name is None or regname is None

        if name:
            self._convert(RemoveConstraint(name))
        else:
            if not regname:
                regname = ".*"
            _rex = re.compile(regname)
            conv_to_do = []
            for name in self.model.constraints:
                if _rex.fullmatch(name):
                    conv_to_do.append(RemoveConstraint(name))
            for c in conv_to_do:
                self._convert(c)
        return self.model

    def eq_to_obj(self, name: str = None, regname: str = None, penalty: float = None):
        assert name is None or regname is None
        if penalty is None:
            penalty = DEFAULT_PENALTY_VALUE

        if name:
            constr = self.model.constraints[name]
            assert isinstance(constr, ConstraintEq)
            self._convert(EqToObj(name, penalty))
        else:
            if not regname:
                regname = ".*"
            _rex = re.compile(regname)
            conv_to_do = []
            for name, constr in self.model.constraints.items():
                if _rex.fullmatch(name) and isinstance(constr, ConstraintEq):
                    conv_to_do.append(EqToObj(name, penalty))
            for c in conv_to_do:
                self._convert(c)
        return self.model

    def int_to_bits(self, mode: str, name: str = None, regname: str = None) -> SympyOpt:
        assert name is None or regname is None

        conv = None
        if mode == "one-hot":
            conv = VarOneHot
        else:
            raise ValueError("Uknown mode {mode}")  # pragma: no cover

        if name:
            intvar = self.model.variables[name]
            assert isinstance(intvar, IntVar)
            self._convert(conv(intvar))
        else:
            if not regname:
                regname = ".*"
            _rex = re.compile(regname)
            conv_to_do = []
            for name, var in self.model.variables.items():
                if _rex.fullmatch(name) and isinstance(var, IntVar):
                    conv_to_do.append(conv(var))
            for c in conv_to_do:
                self._convert(c)
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
