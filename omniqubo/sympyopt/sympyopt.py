from sympy import S

from .constraints import *
from .utils import gen_random_str
from .vars import *

SYMPYOPT_MIN_SENSE = "min"
SYMPYOPT_MAX_SENSE = "max"


class SympyOpt:
    def __init__(self):
        self.constraints = dict()
        self.objective = S(0)
        self.sense = SYMPYOPT_MIN_SENSE
        self.variables = dict()

    def _set_objective(self, obj):
        unknown_vars = _list_unknown_vars(obj, self.variables.keys())
        if len(unknown_vars) != 0:
            return AssertionError(
                f"Variables {unknown_vars} uknown. Use SympyOpt methods to define variables"
            )
        self.objective = obj

    def minimize(self, obj):
        self.sense = SYMPYOPT_MIN_SENSE
        self._set_objective(obj)

    def maximize(self, obj):
        self.sense = SYMPYOPT_MAX_SENSE
        self._set_objective(obj)

    def add_constraint(self, constr: ConstraintAbs, name: str = None):
        if name in self.constraints.keys():
            raise ValueError(f"Constraint {name} already exists")
        if name == None:
            # TODO: (optional) function may check the probability of getting string, and if needed increase the name length
            while name not in self.constraints.keys():
                name = gen_random_str()
        unknown_vars = constr._list_unknown_vars(self.variables.keys())
        if len(unknown_vars) != 0:
            return AssertionError(
                f"Variables {unknown_vars} uknown. Use SympyOpt methods to define variables"
            )
        self.constraints[name] = constr

    def list_constraints(self):
        return self.constraints

    def get_objective(self):
        return self.objective

    def get_constraint(self, name: str):
        if name not in self.constraints.keys():
            raise ValueError(f"Constraint {name} does not exist")
        return self.constraints[name]

    def remove_constraint(self, name: str):
        if name not in self.constraints.keys():
            raise ValueError(f"Constraint {name} does not exist")
        return self.constraints[name].pop(name)

    def get_variable(self, name=None):
        if name == None:
            return self.variables
        else:
            return self.variables[name]

    def get_var(self, name=None):
        return self.get_variables(name)

    def int_var(self, name, lb: int = None, ub: int = None):
        assert ub > lb
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = IntVar(name, lb, ub)
        self.variables[name] = var
        return var.var

    def real_var(self, name, ub: int = None, lb: int = None):
        raise NotImplementedError("Planned at later versions")
        assert ub > lb
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = RealVar(name, lb, ub)
        self.variables[name] = var
        return var.var

    def bit_var(self, name):
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = BitVar(name)
        self.variables[name] = var
        return var.var

    def spin_var(self, name):
        if name in self.variables.keys():
            raise ValueError(f"Variable {name} already exists")
        var = SpinVar(name)
        self.variables[name] = var
        return var.var
