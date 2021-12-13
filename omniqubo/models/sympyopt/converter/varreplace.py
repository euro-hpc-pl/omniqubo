from abc import abstractmethod

import numpy as np
from pandas.core.frame import DataFrame
from sympy import Expr
from sympy.core.evalf import INF

import omniqubo.models.sympyopt as sympyopt
import omniqubo.models.sympyopt.constraints as constraints
import omniqubo.models.sympyopt.vars as vars

from .abs_converter import ConverterSympyOptAbs
from .utils import INTER_STR_SEP


class VarReplace(ConverterSympyOptAbs):
    def __init__(self, var) -> None:
        self.var = var

    @abstractmethod
    def _get_expr_add_constr(self, model: sympyopt.SympyOpt) -> Expr:
        pass

    @abstractmethod
    def _check_model(self, model: sympyopt.SympyOpt) -> None:
        pass

    def _sub_constraint(self, model: sympyopt.SympyOpt, expr: Expr):
        for c in model.constraints.values():
            if isinstance(c, (constraints.ConstraintEq, constraints.ConstraintIneq)):
                c.exprleft = c.exprleft.xreplace({self.var.var: expr})
                c.exprright = c.exprright.xreplace({self.var.var: expr})

    def convert(self, model: sympyopt.SympyOpt) -> sympyopt.SympyOpt:
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
    def __init__(self, var: vars.IntVar) -> None:
        super().__init__(var)

    def interpret(self, sample: DataFrame) -> DataFrame:
        name = self.var.name
        lb = self.var.lb
        ub = self.var.ub

        names = [f"{name}{INTER_STR_SEP}OH_{i}" for i in range(ub - lb + 1)]
        sample["feasible_tmp"] = sample.apply(lambda row: sum(row[n] for n in names) == 1, axis=1)
        expr_comp = (
            lambda row: lb + [row[n] for n in names].index(1) if row["feasible_tmp"] else np.nan
        )
        sample[name] = sample.apply(
            expr_comp,
            axis=1,
        )

        sample["feasible"] &= sample.pop("feasible_tmp")
        for n in names:
            sample.pop(n)

        return sample  # hack to avoid defragmented

    def _get_expr_add_constr(self, model: sympyopt.SympyOpt) -> Expr:
        name = self.var.name
        lb = self.var.lb
        ub = self.var.ub
        xs = [model.bit_var(f"{name}{INTER_STR_SEP}OH_{i}") for i in range(ub - lb + 1)]

        # add constraint
        c = constraints.ConstraintEq(sum(x for x in xs), 1)
        model.add_constraint(c, name=f"{INTER_STR_SEP}OH_{name}")

        # return
        return sum(v * x for x, v in zip(xs, range(lb, ub + 1)))

    def _check_model(self, model: sympyopt.SympyOpt) -> None:
        assert self.var.lb != -INF
        assert self.var.ub != INF


class TrivialIntToBit(VarReplace):
    def __init__(self, var: vars.IntVar) -> None:
        super().__init__(var)

    def interpret(self, sample: DataFrame) -> DataFrame:
        name = f"{self.var.name}{INTER_STR_SEP}itb"
        sample.rename(columns={name: self.var.name})
        return sample

    def _get_expr_add_constr(self, model: sympyopt.SympyOpt) -> Expr:
        var = model.bit_var(f"{self.var.name}{INTER_STR_SEP}itb")
        return self.var.lb + var

    def _check_model(self, model: sympyopt.SympyOpt) -> None:
        assert self.var.ub - self.var.lb == 1


class BitToSpin(VarReplace):
    def __init__(self, var: vars.BitVar, reversed: bool) -> None:
        super().__init__(var)
        self.reversed = reversed

    def interpret(self, sample: DataFrame) -> DataFrame:
        raise NotImplementedError()

    def _get_expr_add_constr(self, model: sympyopt.SympyOpt) -> Expr:
        var = model.spin_var(f"{self.var.name}{INTER_STR_SEP}bts")
        if self.reversed:
            return (1 - var) / 2
        else:
            return (1 + var) / 2

    def _check_model(self, model: sympyopt.SympyOpt) -> None:
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
