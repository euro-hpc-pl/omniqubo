from copy import deepcopy
from typing import Iterable, List

from sympy import Expr, Float, S, preorder_traversal
from sympy.core.function import expand

from .vars import VarAbs


def _list_unknown_vars(obj: Expr, vars: Iterable[str]) -> Iterable:
    return filter(lambda v: v.name not in vars, obj.free_symbols)


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
        if not isinstance(exprleft, Expr):
            exprleft = S(exprleft)
        if not isinstance(exprright, Expr):
            exprright = S(exprright)

        self.exprleft = deepcopy(exprleft)
        self.exprright = deepcopy(exprright)

    def is_eq_constraint(self) -> bool:
        return True

    def __eq__(self, sec: object) -> bool:
        if not isinstance(sec, ConstraintEq):
            return False
        expr1 = expand(self.exprleft - self.exprright)
        expr2 = expand(sec.exprleft - sec.exprright)

        dif = expand(expr1 - expr2)
        for a in preorder_traversal(dif):
            if isinstance(a, Float):
                dif = dif.subs(a, round(a, 15))
        print(dif)
        if dif == 0:
            return True
        dif = expand(expr1 + expr2)
        for a in preorder_traversal(dif):
            if isinstance(a, Float):
                dif = dif.subs(a, round(a, 15))
        print(dif)
        return dif == 0


INEQ_LEQ_SENSE = "leq"
INEQ_GEQ_SENSE = "geq"


class ConstraintIneq(ConstraintAbs):
    def __init__(self, exprleft: Expr, exprright: Expr, sense: str) -> None:
        if not isinstance(exprleft, Expr):
            exprleft = S(exprleft)
        if not isinstance(exprright, Expr):
            exprright = S(exprright)

        self.exprleft = deepcopy(exprleft)
        self.exprright = deepcopy(exprright)
        if sense != INEQ_GEQ_SENSE and sense != INEQ_LEQ_SENSE:
            raise ValueError(f"incorrect sense {sense}")
        self.sense = sense

    def is_ineq_constraint(self) -> bool:
        return True

    def __eq__(self, sec: object) -> bool:
        if not isinstance(sec, ConstraintIneq):
            return False
        same_sense = self.sense == sec.sense

        expr1 = expand(self.exprleft - self.exprright)
        expr2 = expand(sec.exprleft - sec.exprright)
        if same_sense:
            dif = expand(expr1 - expr2)
        else:
            dif = expand(expr1 + expr2)
        print(dif)
        for a in preorder_traversal(dif):
            if isinstance(a, Float):
                dif = dif.subs(a, round(a, 15))
        return dif == 0
