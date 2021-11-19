from abc import abstractmethod

from sympy import Expr
from sympy.core.evalf import INF

from ..sympyopt import SympyOpt
from ..sympyopt.constraints import ConstraintEq, ConstraintIneq
from ..sympyopt.vars import BitVar, IntVar
from .stepconvclass import StepConvAbs
from .utils import INTER_STR_SEP


class VarReplace(StepConvAbs):
    def __init__(self, var) -> None:
        self.var = var

    @abstractmethod
    def _get_expr_add_constr(self, model: SympyOpt) -> Expr:
        pass

    @abstractmethod
    def _check_model(self, model: SympyOpt) -> None:
        pass

    def _sub_constraint(self, model: SympyOpt, expr: Expr):
        for c in model.constraints.values():
            if isinstance(c, (ConstraintEq, ConstraintIneq)):
                c.exprleft = c.exprleft.xreplace({self.var.var: expr})
                c.exprright = c.exprright.xreplace({self.var.var: expr})

    def convert(self, model: SympyOpt) -> SympyOpt:
        name = self.var.name

        self._check_model(model)
        # add variables
        sub_expr = self._get_expr_add_constr(model)

        # substitute expression
        model.objective = model.objective.xreplace({self.var.var: sub_expr})
        self._sub_constraint(model, sub_expr)
        model.variables.pop(name)
        return model


class VarOneHot(VarReplace):
    def __init__(self, var: IntVar) -> None:
        super().__init__(var)

    def interpret(self, sample):
        raise NotImplementedError()

    def _get_expr_add_constr(self, model: SympyOpt) -> Expr:
        name = self.var.name
        lb = self.var.lb
        ub = self.var.ub
        xs = [model.bit_var(f"{name}{INTER_STR_SEP}OH_{i}") for i in range(ub - lb + 1)]

        # add constraint
        c = ConstraintEq(sum(x for x in xs), 1)
        model.add_constraint(c, name=f"{INTER_STR_SEP}OH_{name}")

        # return
        return sum(v * x for x, v in zip(xs, range(lb, ub + 1)))

    def _check_model(self, model: SympyOpt) -> None:
        assert self.var.lb != -INF
        assert self.var.ub != INF


class TrivialIntToBit(VarReplace):
    def __init__(self, var: IntVar) -> None:
        super().__init__(var)

    def interpret(self, sample):
        raise NotImplementedError()

    def _get_expr_add_constr(self, model: SympyOpt) -> Expr:
        var = model.bit_var(f"{self.var.name}{INTER_STR_SEP}itb")
        return self.var.lb + var

    def _check_model(self, model: SympyOpt) -> None:
        assert self.var.ub - self.var.lb == 1


class BitToSpin(VarReplace):
    def __init__(self, var: BitVar, reversed: bool) -> None:
        super().__init__(var)
        self.reversed = reversed

    def interpret(self, sample):
        raise NotImplementedError()

    def _get_expr_add_constr(self, model: SympyOpt) -> Expr:
        var = model.spin_var(f"{self.var.name}{INTER_STR_SEP}bts")
        if self.reversed:
            return (1 - var) / 2
        else:
            return (1 + var) / 2

    def _check_model(self, model: SympyOpt) -> None:
        pass


# class VarUnary(VarReplace):
#     def _get_bits_names(self) -> List[str]:
#         raise NotImplementedError
#         lb, ub = self.var.lb, self.var.ub
#         return [self.var.name + "@{i}" for i in range(lb, ub + 1)]

#     def _get_expression(self) -> Expr:
#         raise NotImplementedError
#         raise self.var.lb + sum(b for b in bits)

#     def _add_constraints(self, model: SympyOpt) -> None:
#         pass
