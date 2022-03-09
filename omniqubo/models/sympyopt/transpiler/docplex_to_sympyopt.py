import sympy
from docplex.mp.basic import Expr
from docplex.mp.constr import ComparisonType, LinearConstraint, QuadraticConstraint
from docplex.mp.dvar import Var
from docplex.mp.linear import ConstantExpr, LinearExpr, MonomialExpr, ZeroExpr
from docplex.mp.model import Model
from docplex.mp.quad import QuadExpr

from omniqubo.transpiler import TranspilerAbs

from ..constraints import INEQ_GEQ_SENSE, ConstraintEq, ConstraintIneq
from ..sympyopt import SympyOpt


class DocplexToSympyopt(TranspilerAbs):
    """Transpiler for transforming Docplex model into SymptOpt model

    Transpiler can transform any quadratic program with quadratic constraints,
    which has bit, integer, or real variables only.
    """

    def _add_constraints(self, model: Model, sympyopt: SympyOpt):
        for cstr in model.iter_constraints():
            if isinstance(cstr, (LinearConstraint, QuadraticConstraint)):
                name = cstr.name
                sense = cstr.sense
                left = self._get_expr(cstr.left_expr, sympyopt)
                right = self._get_expr(cstr.right_expr, sympyopt)
                if sense == ComparisonType.EQ:
                    sympyopt.add_constraint(ConstraintEq(left, right), name=name)
                elif sense == ComparisonType.GE:
                    sympyopt.add_constraint(ConstraintIneq(left, right, INEQ_GEQ_SENSE), name=name)
                elif sense == ComparisonType.LE:
                    sympyopt.add_constraint(ConstraintIneq(left, right), name=name)
                else:
                    ValueError(f"Unknown sense {sense}")  # pragma: no cover
            else:
                ValueError(f"Constraint type {type(cstr)} not implemented")  # pragma: no cover

    def _get_expr(self, obj: Expr, sympyopt: SympyOpt) -> sympy.Expr:
        expr = sympy.S(0)
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
            for (first, sec), val in obj._quadterms.items():
                var1 = sympyopt.get_var(first.name)
                var2 = sympyopt.get_var(sec.name)
                expr += val * var1 * var2
            for var, val in obj._linexpr._terms.items():
                expr += val * sympyopt.get_var(var.name)
            expr += obj._linexpr._constant
        else:
            raise ValueError(f"Unknown objective type {type(obj)}, {obj}")  # pragma: no cover
        return expr

    def _add_objective(self, model: Model, sympyopt: SympyOpt) -> None:
        obj = model.objective_expr
        expr = self._get_expr(obj, sympyopt)
        if model.is_minimized():
            sympyopt.minimize(expr)
        else:
            sympyopt.maximize(expr)

    def _add_variables(self, model: Model, sympyopt: SympyOpt):
        vars = model._vars_by_name
        for name, var in vars.items():
            if var.cplex_typecode == "B":  # bit
                sympyopt.bit_var(name)
            elif var.cplex_typecode == "I":  # integer
                sympyopt.int_var(name, lb=var._lb, ub=var.ub)
            elif var.cplex_typecode == "C":  # continuous
                sympyopt.real_var(name, lb=var._lb, ub=var.ub)
            elif var.cplex_typecode == "S":  # semi-continuous
                raise NotImplementedError("Docplex semi-continuous types not implemented")
            elif var.cplex_typecode == "N":  # semi-integer
                raise NotImplementedError("Docplex semi-integer types not implemented")
            else:
                raise ValueError(f"Unknown cplex_typecode {var.cplex_typecode}")  # pragma: no cover

    def transpile(self, model: Model) -> SympyOpt:
        """Transpile model into SympyOpt model

        :param model: model to be transpiled
        :return: equivalent SympyOpt model
        """
        sympy_model = SympyOpt()
        self._add_variables(model, sympy_model)
        self._add_objective(model, sympy_model)
        self._add_constraints(model, sympy_model)
        return sympy_model

    def can_transpile(self, model: Model) -> bool:
        """Check if model can be transpiled

        Currently equivalent to the fact that all variables are bits, integers,
        or real.

        :type model: model to be transpiled
        :return: flag denoting if model can be transpiled
        """
        for var in model._vars_by_name.values():
            if var.cplex_typecode not in "BCI":
                return False
        return True
