import sympy
from docplex.mp.dvar import Var
from docplex.mp.linear import ConstantExpr, LinearExpr, MonomialExpr, ZeroExpr
from docplex.mp.model import Model
from docplex.mp.quad import QuadExpr
from sympy import S
from sympyopt import SYMPYOPT_MAX_SENSE, SYMPYOPT_MIN_SENSE, SympyOpt

from .converter import ConvertToSymoptAbs


class DocplexToSymoptAbs(ConvertToSymoptAbs):
    def _add_constraints(self, model: Model, sympyopt: SympyOpt):
        raise NotImplementedError

    def _add_objective(self, model: Model, sympyopt: SympyOpt):
        obj = model.objective_expr
        expr = S(0)

        # create expression
        if isinstance(obj, ZeroExpr):
            pass
        elif isinstance(obj, ConstantExpr):
            expr += obj._constant
        elif isinstance(obj, Var):
            expr += sympyopt.get_var(obj.name)
        elif isinstance(obj, MonomialExpr):
            expr += obj._coef * sympyopt.get_var(obj._dvar.name)
        elif isinstance(obj, LinearExpr):
            for var, val in obj._terms.items():
                expr += val * sympyopt.get_var(var.name)
            expr += obj._constant
        elif isinstance(obj, QuadExpr):
            for vars_pair, val in obj._quadterms:
                var1 = sympyopt.get_var(vars_pair.first.name)
                var2 = sympyopt.get_var(vars_pair.second.name)
                expr += val * var1 * var2
            for var, val in obj._linexpr._terms.items():
                expr += val * sympyopt.get_var(var.name)
            expr += obj._linexpr._constant
        else:
            raise ValueError(
                f"New objective type {type(obj)}, {obj}. Please contact authors"
            )

        if model.is_minimized():
            sympyopt.minimize(expr)
        else:
            sympyopt.maximize(expr)
        return sympyopt

    def _add_variables(self, model: Model, sympyopt: SympyOpt):
        vars = model._vars_by_name
        for name, var in vars.items():
            if var.cplex_typecode == "B":  # bit
                sympyopt.bit_var(name)
            elif var.cplex_typecode == "I":  # integer
                if var._lb == 0 and var._ub == 1:  # don't save bit-integers as integers
                    sympyopt.bit_var(name)
                else:
                    sympyopt.int_var(name, lb=var._lb, ub=var.ub)
            elif var.cplex_typecode == "C":  # continuous
                sympyopt.real_var(name, lb=var._lb, ub=var.ub)
            elif var.cplex_typecode == "S":  # semi-continuous
                raise NotImplementedError(
                    "Docplex semi-continuous types not implemented"
                )
            elif var.cplex_typecode == "N":  # semi-integer
                raise NotImplementedError("Docplex semi-integer types not implemented")
            else:
                raise ValueError(
                    f"cplex_typecode {var.cplex_typecode} - please contact authors"
                )

    def convert(self, model: Model) -> SympyOpt:
        sympyopt = SympyOpt()
        self._add_variables(model, sympyopt)
        self._add_objective(model, sympyopt)
        self._add_constraints(model, sympyopt)
        return sympyopt

    def can_convert(self, model: Model) -> bool:
        for var in vars.values():
            if var.cplex_typecode in "SNC":
                return False
        # TODO check the constraints
        return True
