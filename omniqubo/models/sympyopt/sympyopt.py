from __future__ import annotations

from typing import Dict

from sympy import Expr, Integer, S, Symbol, core, expand, total_degree

import omniqubo.utils.utils as utils
from omniqubo.model import MAX_SENSE, MIN_SENSE, ModelAbs

from .constraints import ConstraintAbs, ConstraintEq, ConstraintIneq, _list_unknown_vars
from .utils import _approx_sympy_expr
from .vars import BitVar, IntVar, RealVar, SpinVar, VarAbsSympyOpt


class SympyOpt(ModelAbs):
    """Optimization modeling language based on Sympy.

    The object consist of a dictionary of named constraint, objective function
    which defaults to S(0), sense equal to MIN_SENSE or
    MAX_SENSE and dictionary of variables. The primary use is for
    transforming it into binary or other models.
    """

    def __init__(self) -> None:
        self.constraints: Dict[str, ConstraintAbs] = dict()
        self.objective: Expr = S(0)
        self.sense = MIN_SENSE
        self.variables: Dict[str, VarAbsSympyOpt] = dict()

    def _set_objective(self, obj: Expr) -> None:
        if not isinstance(obj, Expr):
            obj = S(obj)

        unknown_vars = list(_list_unknown_vars(obj, self.variables.keys()))
        if unknown_vars:
            raise ValueError(
                f"Variables {unknown_vars} uknown. Use SympyOpt methods to define variables"
            )
        self.objective = obj

    def minimize(self, obj: Expr) -> None:
        """Set the function to be minimized.

        All variables must be already included in the model.

        :param obj: minimized expression
        :raises ValueError: if variables are not present in the model
        """
        self.sense = MIN_SENSE
        self._set_objective(obj)

    def maximize(self, obj: Expr) -> None:
        """Set the function to be maximized.

        All variables must be already included in the model.

        :param obj: maximized expression
        """
        self.sense = MAX_SENSE
        self._set_objective(obj)

    def add_constraint(self, constraint: ConstraintAbs, name: str = None) -> None:
        """Add constraint to the model.

        If name is not provided, a random string is generated. All variables
        must be already included in the model.

        :param constraint: The constraint
        :param name: name of the constraint, defaults to random name
        :raises ValueError: if variables are not present in the model
        """
        if name in self.constraints.keys():
            raise ValueError(f"Constraint {name} already exists")
        if name is None:
            # HACK: (optional) function may check the probability of getting string,
            # and if needed increase the name length
            name = utils.gen_random_str()
            while name in self.constraints.keys():
                name = utils.gen_random_str()
        unknown_vars = constraint._list_unknown_vars(self.variables.keys())
        if len(unknown_vars) != 0:
            raise AssertionError(
                f"Variables {unknown_vars} uknown. Use SympyOpt methods to define variables"
            )
        self.constraints[name] = constraint

    def list_constraints(self) -> Dict[str, ConstraintAbs]:
        """Return the dictionary of the constraints.

        :return: Dictionary of the constraints.
        """
        return self.constraints

    def get_objective(self) -> Expr:
        """Return the objective function.

        :return: The objective function
        """
        return self.objective

    def get_constraint(self, name: str) -> ConstraintAbs:
        """Return the constraint of the given name.

        :param name: name of the constraint
        :return: the constraint
        """
        return self.constraints[name]

    def get_var(self, name: str) -> Symbol:
        """Return the variable of the given name.

        .. note::
            a Sympy object is returned, not the object of class VarAbs

        :param name: name of the variable
        :return: the variable
        """
        return self.variables[name].var

    def get_vars(self) -> Dict[str, Symbol]:
        """Return the dictionary of variables.

        .. note::
            values of the dictionary are Sympy objects, not the object of class
            VarAbs.

        :return: dictionary of variables
        """
        return {name: var.var for name, var in self.variables.items()}

    def int_var(self, name: str, lb: int = None, ub: int = None) -> Symbol:
        """Create and return integer variable.

        :param name: name of the variable
        :param lb: minimal value, defaults to -INF
        :param ub: maximal value, defaults to INF
        :raises ValueError: if the name is already used
        :return: the Sympy variable
        """
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = IntVar(name, lb, ub)
        self.variables[name] = var
        return var.var

    def real_var(self, name: str, lb: float = None, ub: float = None) -> Symbol:
        """Create and return real variable.

        :param name: name of the variable
        :param lb: minimal value, defaults to -INF
        :param ub: maximal value, defaults to INF
        :raises ValueError: if the name is already used
        :return: the Sympy variable
        """
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = RealVar(name, lb, ub)
        self.variables[name] = var
        return var.var

    def bit_var(self, name: str) -> Symbol:
        """Create and return binary variable.

        :param name: name of the variable
        :raises ValueError: if the name is already used
        :return: the Sympy variable
        """
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = BitVar(name)
        self.variables[name] = var
        return var.var

    def spin_var(self, name: str) -> Symbol:
        """Create and return spin variable

        :param name: name of the variable
        :raises ValueError: if the name is already used
        :return: the Sympy variable
        """
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = SpinVar(name)
        self.variables[name] = var
        return var.var

    def __eq__(self, model2) -> bool:
        """Check if two optimization models equal.

        Equality is equivalent to: same sense, approximately same objective
        function, same constraints list, and same variables.
        """
        if self.sense != model2.sense:
            return False
        if _approx_sympy_expr(expand(self.objective - model2.objective)) != 0:
            return False
        if self.constraints.keys() != model2.constraints.keys():
            return False
        for k in self.constraints:
            if not self.constraints[k] == model2.constraints[k]:
                return False
        if self.variables.keys() != model2.variables.keys():
            return False
        for k in self.variables:
            if not self.variables[k] == model2.variables[k]:
                return False
        return True

    def __str__(self) -> str:
        out_string = "SympyOpt instance\n"
        out_string += "minimize:\n" if self.sense == MIN_SENSE else "maximize\n"
        out_string += f"   {self.objective}\n"
        if self.constraints:
            out_string += "such that:\n"
            for name in self.constraints:
                out_string += f"   {name}: {self.constraints[name]}\n"
        if self.variables:
            out_string += "variables:\n"
            for name in self.variables:
                out_string += f"   {name}: {self.variables[name]}\n"
        return out_string

    def _bitspin_simp_rec(self, expr: Expr) -> Expr:
        if expr.is_number:
            return expr
        if isinstance(expr, core.symbol.Symbol):
            return expr
        if isinstance(expr, core.power.Pow):
            if isinstance(expr.exp, Integer) and expr.exp > 0 and isinstance(expr.base, Symbol):
                name = expr.base.name
                var = self.variables[name]
                if isinstance(var, BitVar):
                    return expr.base
                elif isinstance(var, SpinVar):
                    if expr.exp % 2 == 0:
                        return S(1)
                    else:
                        return expr.base
                else:
                    return expr
        if isinstance(expr, core.mul.Mul):  # if produce
            tmp_term = S(1)
            for el_prod in expr._args:
                # below there is recursive run, but can be only once if expr is expanded polynomial
                tmp_term *= self._bitspin_simp_rec(el_prod)
            return tmp_term
        # don't do anything inside a non-polynomial parts, for example exp(b**3) == exp(b**3)
        return expr

    def _bitspin_simp(self, expr: Expr) -> Expr:
        expr = expand(expr)
        if isinstance(expr, core.add.Add):
            new_expr = S(0)
            for el in expr._args:
                new_expr += self._bitspin_simp_rec(el)
            return new_expr
        else:
            return self._bitspin_simp_rec(expr)

    def _are_constrs_poly(self, order=None) -> bool:
        for c in self.constraints.values():
            if not isinstance(c, (ConstraintEq, ConstraintIneq)):
                return False
            for expr in [c.exprleft, c.exprright]:
                if not expr.is_polynomial():
                    return False
                if order is not None:
                    if total_degree(self._bitspin_simp(expr)) > order:
                        return False
        return True

    def is_lip(self) -> bool:
        """Check if model is Linear Integer Program.

        Model is Linear Integer Program if all variables are BitVar or IntVar,
        and objective and constraints are linear

        :return: flag stating if the model is Linear Integer Program
        """
        if not all(isinstance(v, (BitVar, IntVar)) for v in self.variables.values()):
            return False
        if not self.objective.is_polynomial():
            return False
        objective_copy = self._bitspin_simp(self.objective)
        if total_degree(objective_copy) > 1:
            return False
        if not self._are_constrs_poly(order=1):
            return False
        return True

    def is_qip(self) -> bool:
        """Check if model is Quadratic Integer Program.

        Model is Quadratic Integer Program if all variables are BitVar or
        IntVar,  objective is quadratic polynomial, and constraints are linear.

        :return: flag stating if the model is Quadratic Integer Program
        """
        if not all(isinstance(v, (BitVar, IntVar)) for v in self.variables.values()):
            return False
        if not self.objective.is_polynomial():
            return False
        objective_copy = self._bitspin_simp(self.objective)
        if total_degree(objective_copy) > 2:
            return False
        if not self._are_constrs_poly(order=1):
            return False
        return True

        raise NotImplementedError()

    def is_pip(self) -> bool:
        """Check if model is Polynomial Integer Program.

        Model is Polynomial Integer Program if all variables are BitVar or
        IntVar, objective and constraints are polynomials.

        :return: flag stating if the model is Polynomial Integer Program
        """
        if not all(isinstance(v, (BitVar, IntVar)) for v in self.variables.values()):
            return False
        if not self.objective.is_polynomial():
            return False
        if not self._are_constrs_poly():
            return False
        return True

    def is_qcqp(self) -> bool:
        """Check if model is Quadratically Constrained Quadratic Program.

        Model is Quadratically Constrained Quadratic Program if all variables
        are BitVar or IntVar, objective and constraints are quadratic
        polynomials.

        :return: flag stating if the model is Quadratically Constrained
            Quadratic Program
        """
        if not all(isinstance(v, (BitVar, IntVar)) for v in self.variables.values()):
            return False
        if not self.objective.is_polynomial():
            return False
        objective_copy = self._bitspin_simp(self.objective)
        if total_degree(objective_copy) > 2:
            return False
        if not self._are_constrs_poly(order=2):
            return False
        return True

    def is_bm(self) -> bool:
        """Check if model is Binary Model.

        Model is Binary Model if all variables are BitVar or SpinVar.

        :return: flag stating if the model is Binary Model
        """
        vars = self.variables.values()
        return all(isinstance(v, (BitVar, SpinVar)) for v in vars)

    def is_qubo(self) -> bool:
        """Check if model is Quadratic Unconstrained Binary Optimization.

        Model is Quadratic Unconstrained Binary Optimization if all variables
        are BitVar, objective function is quadratic polynomial and there are no
        constraints.

        :return: flag stating if the model is Quadratic Unconstrained Binary
            Optimization
        """
        if len(self.list_constraints()) > 0:
            return False
        if not all(isinstance(v, BitVar) for v in self.variables.values()):
            return False
        if not self.objective.is_polynomial():
            return False
        objective_copy = self._bitspin_simp(self.objective)
        return total_degree(objective_copy) <= 2

    def is_ising(self, locality: int = None) -> bool:
        """Check if model is an Ising Model.

        Model is Ising model if all variables are SpinVar, objective function is
        a polynomial of at most locality order and there are no constraints.

        :param locality: maximal locality, defaults to 2
        :return: flag stating if the model is Quadratic Unconstrained Binary
            Optimization
        """

        if locality is None:
            locality = 2
        assert locality > 0
        if len(self.list_constraints()) > 0:
            return False
        if not all(isinstance(v, SpinVar) for v in self.variables.values()):
            return False
        if not self.objective.is_polynomial():
            return False
        objective_copy = self._bitspin_simp(self.objective)
        return total_degree(objective_copy) <= locality

    def is_hobo(self) -> bool:
        """Check if model is Higher Order Binary Optimization.

        Model is Higher Order Binary Optimization if all variables are BitVar,
        objective function is a polynomial and there are no constraints.

        :return: flag stating if the model is Higher Order Binary Optimization
        """
        if len(self.list_constraints()) > 0:
            return False
        if not all(isinstance(v, BitVar) for v in self.variables.values()):
            return False
        return self.objective.is_polynomial()
