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
    """Replaces the variable with some formula.

    Transform all occurrences of the variable var in the model with the binary
    formula, and add extra constraint if required. This is an abstract class
    which can be used for various integer encodings.

    .. note::
        Variable var should disappear from the model, including its list of
        variables.

    :param var: variable to be replaced
    """

    def __init__(self, var) -> None:
        self.var = var

    @abstractmethod
    def _get_expr_add_constr(self, model: sympyopt.SympyOpt) -> Expr:
        """creates the formula for var and adds constraints to model.

        .. note::
            model should be updated with the constraints required for the
            transformation

        :param model: model to be transformed, with new constraints added
        :return: expression for replacing all occurrences of var
        """
        pass

    @abstractmethod
    def _check_model(self, model: sympyopt.SympyOpt) -> None:
        """Checks if the model and variables are allowed to be transformed.

        :param model: the model to be transformed
        """
        pass

    def _sub_constraint(self, model: sympyopt.SympyOpt, expr: Expr):
        for c in model.constraints.values():
            if isinstance(c, (constraints.ConstraintEq, constraints.ConstraintIneq)):
                c.exprleft = c.exprleft.xreplace({self.var.var: expr})
                c.exprright = c.exprright.xreplace({self.var.var: expr})

    def convert(self, model: sympyopt.SympyOpt) -> sympyopt.SympyOpt:
        """replaces all occurrences for self.var and add constraints.

        :param model: model to be transformed
        :return: transformed model
        """
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
    """Replace integer variables with one-hot encoding.

    Replaces integer variables with one-hot encoding, and add constraint that
    sum of new added bits is equal to one. For variable lb <= y <= ub
    the encoding creates ub-lb+1 binary variables. The limits of y needs to be
    finite integer numbers.

    :param var: the replaced integer variable
    """

    def __init__(self, var: vars.IntVar) -> None:
        super().__init__(var)

    def interpret(self, samples: DataFrame) -> DataFrame:
        """Compute the value of the previously removed integer variable.

        Also remove corresponding binary variables and update feasibility
        record. If the sample does not meet the condition, np.nan is placed

        :param samples: optimized samples
        :return: updated samples with values of the replaced integer
        """
        name = self.var.name
        lb = self.var.lb
        ub = self.var.ub

        names = [f"{name}{INTER_STR_SEP}OH_{i}" for i in range(ub - lb + 1)]
        samples["feasible_tmp"] = samples.apply(lambda row: sum(row[n] for n in names) == 1, axis=1)
        expr_comp = (
            lambda row: lb + [row[n] for n in names].index(1) if row["feasible_tmp"] else np.nan
        )
        samples[name] = samples.apply(
            expr_comp,
            axis=1,
        )

        samples["feasible"] &= samples.pop("feasible_tmp")
        for n in names:
            samples.pop(n)

        return samples

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
    """Replace integer with binary variable.

    Replaces integer variables y with binary variable lb + b, where
    lb <= y <= lb+1 is assumed. lb should be finite integer number.

    :param var: the replaced integer variable
    """

    def __init__(self, var: vars.IntVar) -> None:
        super().__init__(var)

    def interpret(self, samples: DataFrame) -> DataFrame:
        """Compute the value of the previously removed integer variable.

        Also remove corresponding binary variables.

        :param samples: optimized samples
        :return: updated samples with values of the replaced integer
        """
        name = f"{self.var.name}{INTER_STR_SEP}itb"
        samples.rename(columns={name: self.var.name})
        samples[self.var.name] = samples[self.var.name] + self.var.lb
        return samples

    def _get_expr_add_constr(self, model: sympyopt.SympyOpt) -> Expr:
        var = model.bit_var(f"{self.var.name}{INTER_STR_SEP}itb")
        return self.var.lb + var

    def _check_model(self, model: sympyopt.SympyOpt) -> None:
        assert self.var.ub - self.var.lb == 1


class BitToSpin(VarReplace):
    """Replace binary variable with spin variable.

    Replaces binary variable b with spin variable s. The formula is (1-s)/2
    if reversed is set to true, or (1+s)/2 otherwise.

    :param var: the replaced binary variable
    :param reversed: the flag denoting which formula is used for replacement
    """

    def __init__(self, var: vars.BitVar, reversed: bool) -> None:
        super().__init__(var)
        self.reversed = reversed

    def interpret(self, samples: DataFrame) -> DataFrame:
        """Compute the value of the previously removed binary variable.

        Also remove corresponding spin variable.

        :param samples: optimized samples
        :return: updated samples with values of the replaced binary variable
        """
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
