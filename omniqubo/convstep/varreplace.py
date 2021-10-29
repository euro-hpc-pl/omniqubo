from sympy.core.evalf import INF
from sympyopt import SympyOpt
from sympyopt.constraints import ConstraintEq

from .stepconvclass import StepConvAbs


class VarReplace(StepConvAbs):
    def __init__(self, var: str):
        if var.lb:
            assert var.lb > -INF
        if var.ub:
            assert var.ub < INF
        self.var = var

    def _get_bits_names(self):
        raise NotImplementedError()

    def _get_expression(self, bits=None):
        raise NotImplementedError()

    def _add_constraints(self, model):
        # Keep the error as lack of constraints is rare
        raise NotImplementedError()

    def convert(self, model: SympyOpt):
        name = self.var.name

        # remove and add variables
        model.variables.pop(name)
        bit_names = self._get_bits_names()
        bits = []
        for bname in bit_names:
            bit = model.bit_var(bname)
            bits.append(bit)

        # substitute expression
        expr = self._get_expression(bits)
        model.objective.subs(self.var.var, expr)
        for cname in model.list_constraints:
            model.constraints[cname].subs(self.var.var, expr)

        # add constraints
        self._add_constraints(model, bits)
        return model


class VarOneHot(VarReplace):
    def _get_bits_names(self):
        lb, ub = self.var.lb, self.var.ub
        return [self.var.name + "@{i}" for i in range(lb, ub + 1)]

    def _get_expression(self, bits=None):
        lb, ub = self.var.lb, self.var.ub
        raise sum(val * b for b, val in zip(bits, range(lb, ub + 1)))

    def _add_constraints(self, model: SympyOpt, bits):
        c = ConstraintEq(sum(bits), 1)
        model.add_constraint(c, self.varname + "_onehot")


class VarUnary(VarReplace):
    def _get_bits_names(self):
        lb, ub = self.var.lb, self.var.ub
        return [self.var.name + "@{i}" for i in range(lb, ub + 1)]

    def _get_expression(self, bits=None):
        raise self.var.lb + sum(b for b in bits)

    def _add_constraints(self, model: SympyOpt, bits):
        pass
