import re
from copy import deepcopy
from typing import List

from sympy import S, Symbol, core, expand, total_degree

from ..convstep import StepConvAbs, VarOneHot
from ..soptconv import convert_to_sympyopt
from ..sympyopt import BitVar, SpinVar, SympyOpt
from ..sympyopt.vars import IntVar


class Omniqubo:
    def __init__(self, model, verbatim_logs: bool = False) -> None:
        self.orig_model = deepcopy(model)
        self.model = convert_to_sympyopt(self.orig_model)  # type: SympyOpt
        self.logs = []  # type: List[StepConvAbs]
        self.model_logs = []  # type: List[SympyOpt]
        self.verbatim_logs = verbatim_logs

    def _convert(self, convstep):
        self.logs.append(convstep)
        self.model = convstep.convert(self.model)
        if self.verbatim_logs:
            self.model_logs.append(deepcopy(self.model))
        return self.model

    def interpret(self, samples, general_form=True):
        raise NotImplementedError()

    def to_qubo(self):
        raise NotImplementedError()

    def to_hobo(self):
        raise NotImplementedError()

    def export(self, mode: str):
        raise NotImplementedError()

    def _bitspin_polysimp_rec(self, expr, vars=None, mode="bit"):
        if expr.is_number:
            return expr
        if isinstance(expr, core.symbol.Symbol):
            return expr
        if isinstance(expr, core.power.Pow):
            assert isinstance(expr.exp, int) or isinstance(expr.exp, Symbol)
            varname = expr.base.__name__
            var = self.model.variables[varname]
            if (
                vars is None
                and not isinstance(var, BitVar)
                and not isinstance(var, SpinVar)
            ):
                return expr
            if vars is not None and var not in vars:
                return expr

            if mode == "bit":
                return expr.base
            else:  # must be spin
                if expr.exp % 2 == 0:
                    return S(1)
                else:
                    return expr.base
        if isinstance(expr, core.mul.Mul):  # if produce
            tmp_term = S(1)
            for el_prod in expr._args:
                # below there is recursive run, but can be only once if expr is expanded polynomial
                tmp_term *= self._bitspin_polysimp_rec(el_prod, vars, mode)
            return tmp_term
        raise "Unrecognized operation, contact authors the code"

    def _bitspin_polysimp(self, expr, vars=None, mode="bit"):
        assert mode == "bit" or mode == "spin"
        # function works only for polynomials.
        # Possible waste of time if we can assume the expression is polynomial
        assert expr.is_polynomial()

        expr = expand(expr)
        if isinstance(expr, core.add.Add):
            # not a monomial
            new_expr = S(0)
            for el in expr._args:
                new_expr += self._bitspin_polysimp_rec(el, vars, mode)
            return new_expr
        else:
            # monomial
            return self._bitspin_polysimp_rec(expr, vars, mode)

    def is_qubo(self):
        if len(self.model.list_constraints()) > 0:
            return False
        if not all(isinstance(v, BitVar) for v in self.model.variables):
            return False
        if not self.model.objective.is_polynomial():
            return False
        objective_copy = self._bitspin_polysimp(self.mode.objective)
        return total_degree(objective_copy) <= 2

    def is_hobo(self):
        if len(self.model.list_constraints()) > 0:
            return False
        if not all(isinstance(v, BitVar) for v in self.model.variables):
            return False
        return self.model.objective.is_polynomial()

    def int_to_bits(self, mode: str, name: str = None, regname: str = None):
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

    def is_lip(self):
        raise NotImplementedError()

    def is_qip(self):
        raise NotImplementedError()

    def is_qcqp(self):
        raise NotImplementedError()

    def is_bm(self):
        vars = self.model.variables.values()
        return all(isinstance(v, (BitVar, SpinVar)) for v in vars)

    def is_mip(self):
        raise NotImplementedError()
