from copy import deepcopy
from typing import List

from convstep import StepConvAbs
from soptconv import convert_to_sympyopt
from sympy import S, Symbol, core, expand, total_degree
from sympyopt import BitVar, SpinVar, SympyOpt


class Omniqubo:
    def __init__(self, model, verbatim_logs=False) -> None:
        self.orig_model = deepcopy(model)
        self.model = convert_to_sympyopt(self.orig_model)
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
        if not all(isinstance(v, BitVar) for v in self.variables):
            return False
        if not self.model.objective.is_polynomial():
            return False
        self.model.objective = self._bitspin_polysimp(self.mode.objective)
        return total_degree(self.model.objective) <= 2

    def is_hobo(self):
        if len(self.model.list_constraints()) > 0:
            return False
        if not all(isinstance(v, BitVar) for v in self.variables):
            return False
        return self.model.objective.is_polynomial()

    def is_linear_integer_programming(self):
        raise NotImplementedError()

    def is_lip(self):
        return self.is_integer_linear_programming()

    def is_quadratic_integer_programming(self):
        raise NotImplementedError()

    def is_qip(self):
        return self.is_quadratic_integer_programming()

    def is_constrained_quadratic_program(self):
        raise NotImplementedError()

    def is_cqm(self):
        return self.is_constrained_quadratic_program()

    def is_binary_quadratic_model(self):
        raise NotImplementedError()

    def is_bqm(self):
        return self.is_binary_quadratic_model()

    def is_mixed_integer_program(self):
        raise NotImplementedError()

    def is_mip(self):
        return self.is_mixed_integer_program()
