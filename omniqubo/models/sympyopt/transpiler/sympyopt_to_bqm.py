from typing import Dict

import dimod
from sympy import Expr, core

from omniqubo.models.transpiler import TransiplerAbs

from ..sympyopt import SYMPYOPT_MIN_SENSE, SympyOpt


class SympyOptToDimod(TransiplerAbs):
    def __init__(self, mode=None) -> None:
        if mode is None:
            mode = "bqm"
        assert mode == "bqm"
        self.mode = mode

    def _convert_monomial(self, expr: Expr, linear: Dict, quadratic: Dict) -> float:
        # assumes expr is expanded and simplified
        if expr.is_number:
            return float(expr)
        if isinstance(expr, core.symbol.Symbol):
            linear[expr.name] = 1.0
        if isinstance(expr, core.mul.Mul):
            if len(expr._args) == 2:
                if expr._args[0].is_number:
                    linear[expr._args[1].name] = expr._args[0]
                else:
                    quadratic[(expr._args[0].name, expr._args[1].name)] = 1.0
            if len(expr._args) == 3:
                quadratic[(expr._args[1].name, expr._args[2].name)] = expr._args[0]
        return 0.0

    def convert(self, model: SympyOpt) -> dimod.BinaryQuadraticModel:
        assert self.can_convert(model)
        obj = model.objective
        obj = model._bitspin_simp(obj)
        vartype = dimod.BINARY
        linear = {}  # type: Dict
        quadratic = {}  # type: Dict
        offset = 0.0

        if isinstance(obj, core.add.Add):
            for el in obj._args:
                offset += self._convert_monomial(el, linear, quadratic)
        else:
            offset += self._convert_monomial(obj, linear, quadratic)
        return dimod.BinaryQuadraticModel(linear, quadratic, offset=offset, vartype=vartype)

    def can_convert(self, model: SympyOpt) -> bool:
        return model.is_qubo() and model.sense == SYMPYOPT_MIN_SENSE
