from copy import deepcopy
from typing import Iterable, List

from sympy import Expr, S

from .vars import VarAbs


def _list_unknown_vars(obj: Expr, vars: Iterable[str]) -> Iterable:
    return filter(lambda v: v not in vars, obj.free_symbols)


class ConstraintAbs:
    def __init__(self) -> None:
        self.exprleft = S(0)
        self.exprright = S(0)
        pass

    def is_eq_constraint(self) -> bool:
        return False

    def is_ineq_constraint(self) -> bool:
        return False

    def is_type_constraint(self) -> bool:
        return False

    def _list_unknown_vars(self, vars: Iterable[str]) -> List[VarAbs]:
        lvars_uknown = _list_unknown_vars(self.exprleft, vars)
        rvars_uknown = _list_unknown_vars(self.exprleft, vars)
        return list(lvars_uknown) + list(rvars_uknown)


class ConstraintEq(ConstraintAbs):
    def __init__(self, exprleft: Expr, exprright: Expr) -> None:
        self.exprleft = deepcopy(exprleft)
        self.exprright = deepcopy(exprright)

    def is_eq_constraint(self) -> bool:
        return True


INEQ_LEQ_SENSE = "leq"
INEQ_GEQ_SENSE = "geq"


class ConstraintIneq(ConstraintAbs):
    def __init__(self, exprleft: Expr, exprright: Expr, sense: str) -> None:
        self.exprleft = deepcopy(exprleft)
        self.exprright = deepcopy(exprright)
        if sense != INEQ_GEQ_SENSE and sense != INEQ_LEQ_SENSE:
            raise ValueError(f"incorrect sense {sense}")
        self.sense = sense

    def is_ineq_constraint(self) -> bool:
        return True
