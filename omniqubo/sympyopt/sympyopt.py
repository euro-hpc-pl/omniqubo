from typing import Dict

from sympy import Expr, S, Symbol, expand
from sympy.core.evalf import INF

from .constraints import ConstraintAbs, _list_unknown_vars
from .utils import gen_random_str
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
            while name not in self.constraints.keys():
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

    def int_var(self, name: str, lb: int = None, ub: int = None) -> IntVar:
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

    def real_var(self, name: str, lb: float = None, ub: float = None) -> RealVar:
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

    def bit_var(self, name: str) -> BitVar:
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = BitVar(name)
        self.variables[name] = var
        return var.var

    def spin_var(self, name: str) -> SpinVar:
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = SpinVar(name)
        self.variables[name] = var
        return var.var

    def __eq__(self, model2) -> bool:
        if self.sense != model2.sense:
            return False
        if expand(self.objective - model2.objective) != 0:
            return False
        if self.constraints.keys() != model2.constraints.keys():
            return False
        for k in self.constraints:
            if not self.constraints[k] == model2.constraints[k]:
                return False
        if self.variables.keys() != model2.variables.keys():
            return False
        # TODO "compare variables as well"
        return True
