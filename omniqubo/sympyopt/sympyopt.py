from typing import Dict

from sympy import Expr, Integer, S, Symbol, core, expand, total_degree
from sympy.core.evalf import INF

from .constraints import ConstraintAbs, ConstraintEq, ConstraintIneq, _list_unknown_vars
from .utils import _approx_sympy_expr, gen_random_str
from .vars import BitVar, IntVar, RealVar, SpinVar, VarAbs

SYMPYOPT_MIN_SENSE = "min"
SYMPYOPT_MAX_SENSE = "max"


class SympyOpt:
    def __init__(self) -> None:
        self.constraints = dict()  # type: Dict[str,ConstraintAbs]
        self.objective = S(0)
        self.sense = SYMPYOPT_MIN_SENSE
        self.variables = dict()  # type: Dict[str,VarAbs]

    def _set_objective(self, obj: Expr) -> None:
        if not isinstance(obj, Expr):
            obj = S(obj)

        unknown_vars = list(_list_unknown_vars(obj, self.variables.keys()))
        if unknown_vars:
            raise AssertionError(
                f"Variables {unknown_vars} uknown. Use SympyOpt methods to define variables"
            )
        self.objective = obj

    def minimize(self, obj: Expr) -> None:
        self.sense = SYMPYOPT_MIN_SENSE
        self._set_objective(obj)

    def maximize(self, obj: Expr) -> None:
        self.sense = SYMPYOPT_MAX_SENSE
        self._set_objective(obj)

    def add_constraint(self, constr: ConstraintAbs, name: str = None) -> None:
        if name in self.constraints.keys():
            raise ValueError(f"Constraint {name} already exists")
        if name is None:
            # HACK: (optional) function may check the probability of getting string,
            # and if needed increase the name length
            name = gen_random_str()
            while name in self.constraints.keys():
                name = gen_random_str()
        unknown_vars = constr._list_unknown_vars(self.variables.keys())
        if len(unknown_vars) != 0:
            raise AssertionError(
                f"Variables {unknown_vars} uknown. Use SympyOpt methods to define variables"
            )
        self.constraints[name] = constr

    def list_constraints(self) -> Dict[str, ConstraintAbs]:
        return self.constraints

    def get_objective(self) -> Expr:
        return self.objective

    def get_constraint(self, name: str) -> ConstraintAbs:
        if name not in self.constraints.keys():
            raise ValueError(f"Constraint {name} does not exist")
        return self.constraints[name]

    def remove_constraint(self, name: str) -> ConstraintAbs:
        if name not in self.constraints.keys():
            raise ValueError(f"Constraint {name} does not exist")
        return self.constraints.pop(name)

    def get_var(self, name: str) -> Symbol:
        return self.variables[name].var

    def get_vars(self) -> Dict[str, Symbol]:
        return {name: var.var for name, var in self.variables.items()}

    def int_var(self, name: str, lb: int = None, ub: int = None) -> Symbol:
        if lb is None:
            lb = -INF
        if ub is None:
            ub = -INF
        assert lb < ub
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = IntVar(name, lb, ub)
        self.variables[name] = var
        return var.var

    def real_var(self, name: str, lb: float = None, ub: float = None) -> Symbol:
        if lb is None:
            lb = -INF
        if ub is None:
            ub = -INF
        raise NotImplementedError("Planned at later versions")
        assert ub > lb
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = RealVar(name, lb, ub)
        self.variables[name] = var
        return var.var

    def bit_var(self, name: str) -> Symbol:
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = BitVar(name)
        self.variables[name] = var
        return var.var

    def spin_var(self, name: str) -> Symbol:
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = SpinVar(name)
        self.variables[name] = var
        return var.var

    def __eq__(self, model2) -> bool:
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
        out_string += "minimize:\n" if self.sense == SYMPYOPT_MIN_SENSE else "maximize\n"
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

    def is_pp(self) -> bool:
        if not all(isinstance(v, (BitVar, IntVar)) for v in self.variables.values()):
            return False
        if not self.objective.is_polynomial():
            return False
        if not self._are_constrs_poly():
            return False
        return True

    def is_qcqp(self) -> bool:
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
        vars = self.variables.values()
        return all(isinstance(v, (BitVar, SpinVar)) for v in vars)

    def is_qubo(self) -> bool:
        if len(self.list_constraints()) > 0:
            return False
        if not all(isinstance(v, BitVar) for v in self.variables.values()):
            return False
        if not self.objective.is_polynomial():
            return False
        objective_copy = self._bitspin_simp(self.objective)
        return total_degree(objective_copy) <= 2

    def is_ising(self, locality: int = None) -> bool:
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
        if len(self.list_constraints()) > 0:
            return False
        if not all(isinstance(v, BitVar) for v in self.variables.values()):
            return False
        return self.objective.is_polynomial()
