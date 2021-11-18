import re
from copy import deepcopy
from typing import List

from omniqubo.sympyopt.constraints import ConstraintEq

from ..convstep import EqToObj, StepConvAbs, VarOneHot
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
            raise ValueError("Uknown mode {mode}")

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

    def is_qubo(self):
        return self.model.is_qubo()

    def is_hobo(self):
        return self.model.is_hobo()

    def is_lip(self):
        return self.model.is_lip()

    def is_qip(self):
        return self.model.is_qip()

    def is_qcqp(self):
        return self.model.is_qcqp()

    def is_bm(self):
        return self.model.is_bm()
