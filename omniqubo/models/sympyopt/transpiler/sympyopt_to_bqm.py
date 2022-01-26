from typing import Dict

import dimod
from sympy import Expr, core

from omniqubo.transpiler import TransiplerAbs

from ..sympyopt import MIN_SENSE, SympyOpt


class SympyOptToDimod(TransiplerAbs):
    """Transpile SympyOpt model into Dimod object

    At the moment mode can only be None or "bqm", both resulting in returning
    dimod.BinaryQuadraticModel. Transpiler assumes the output model is a QUBO
    and it is minimization problem.

    :param mode: type of the model returned by transpile
    """

    def __init__(self, mode: str = None) -> None:
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

    def transpile(self, model: SympyOpt) -> dimod.BinaryQuadraticModel:
        """Transpile SympyOpt model into BinaryQuadraticModel

        :param model: model to be transpiled
        :return: newly constructed model
        """
        assert self.can_transpile(model)
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

    def can_transpile(self, model: SympyOpt) -> bool:
        """Check if SympyOpt can be transpiled

        Currently equivalent to the fact that SympyOpt is minimization problem
        and QUBO.

        :param model: checked model
        :return: flag denoting if model can be transpiled
        """
        return model.is_qubo() and model.sense == MIN_SENSE
