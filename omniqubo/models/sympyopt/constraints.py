from copy import deepcopy
from typing import Iterable, List

from sympy import Expr, Float, S, preorder_traversal
from sympy.core.function import expand

from omniqubo.constraints import ConstraintAbs

from .utils import _approx_sympy_expr
from .vars import VarAbs


def _list_unknown_vars(obj: Expr, vars: Iterable[str]) -> Iterable:
    return filter(lambda v: v.name not in vars, obj.free_symbols)


class ConstraintSympyopt(ConstraintAbs):
    def __init__(self) -> None:
        pass


class ConstraintEq(ConstraintSympyopt):
    """Equality constraint for SympyOpt

    Constraint of the form exprleft == exprright.

    :param exprleft: The left expression of the equality
    :param exprright: The right expression of the equality
    """

    def __init__(self, exprleft: Expr, exprright: Expr) -> None:
        if not isinstance(exprleft, Expr):
            exprleft = S(exprleft)
        if not isinstance(exprright, Expr):
            exprright = S(exprright)

        self.exprleft = deepcopy(exprleft)
        self.exprright = deepcopy(exprright)

    def is_eq_constraint(self) -> bool:
        """Check if the constraint is an equality

        :return: True
        """
        return True

    def is_ineq_constraint(self) -> bool:
        """Check if the constraint is an inequality

        :return: False
        """
        return False

    def __eq__(self, sec: object) -> bool:
        """Compares two constraints

        The reference object needs to be ConstraintEq. f == g and
        a == b are equivalent if (f - g) - (a - b) or (f - g) + (a - b) is
        zero. The expressions should be sufficiently simple so that SymPy can
        easily simplify them. Numbers appearing in the expressions are
        approximated.
        """
        if not isinstance(sec, ConstraintEq):
            return False
        expr1 = expand(self.exprleft - self.exprright)
        expr2 = expand(sec.exprleft - sec.exprright)

        dif = expand(expr1 - expr2)
        for a in preorder_traversal(dif):
            if isinstance(a, Float):
                dif = dif.xreplace({a: round(a, 15)})
        if dif == 0:
            return True
        dif = expand(expr1 + expr2)
        for a in preorder_traversal(dif):
            if isinstance(a, Float):
                dif = dif.xreplace({a: round(a, 15)})
        return dif == 0

    def __str__(self) -> str:
        return f"{self.exprleft} == {self.exprright}"

    def _list_unknown_vars(self, vars: Iterable[str]) -> List[VarAbs]:
        lvars_uknown = _list_unknown_vars(self.exprleft, vars)
        rvars_uknown = _list_unknown_vars(self.exprleft, vars)
        return list(lvars_uknown) + list(rvars_uknown)


INEQ_LEQ_SENSE = "leq"
INEQ_GEQ_SENSE = "geq"


class ConstraintIneq(ConstraintSympyopt):
    """Inequality constraint for SympyOpt

    Constraint of the form exprleft <= exprright for sense equal to
    INEQ_LEQ_SENSE or exprleft >= exprright for sense equal to INEQ_GEQ_SENSE.

    :param exprleft: The left expression of the inequality
    :param exprright: The right expression of the inequality
    :param sense: the sense of the inequality
    """

    def __init__(self, exprleft: Expr, exprright: Expr, sense: str = None) -> None:
        if not isinstance(exprleft, Expr):
            exprleft = S(exprleft)
        if not isinstance(exprright, Expr):
            exprright = S(exprright)
        if sense is None:
            sense = INEQ_LEQ_SENSE

        self.exprleft = deepcopy(exprleft)
        self.exprright = deepcopy(exprright)
        if sense != INEQ_GEQ_SENSE and sense != INEQ_LEQ_SENSE:
            raise ValueError(f"incorrect sense {sense}")
        self.sense = sense

    def is_eq_constraint(self) -> bool:
        """Check if the constraint is an equality

        :return: False
        """
        return False

    def is_ineq_constraint(self) -> bool:
        """Check if the constraint is an inequality

        :return: True
        """
        return True

    def __eq__(self, sec: object) -> bool:
        """Compares two inequality constraints

        The reference object needs to be ConstraintIneq. If senses are the same
        for both inequalities, for example f >= g and
        a >= b, then constraints are equivalent if (f - g) - (a - b) == 0.
        Otherwise the condition is (f - g) + (a - b) >= 0 is approximately zero.
        The expressions should be sufficiently simple so that SymPy can easily
        simplify them. Numbers appearing in the expressions are approximated.
        """
        if not isinstance(sec, ConstraintIneq):
            return False
        same_sense = self.sense == sec.sense

        expr1 = expand(self.exprleft - self.exprright)
        expr2 = expand(sec.exprleft - sec.exprright)
        if same_sense:
            dif = expand(expr1 - expr2)
        else:
            dif = expand(expr1 + expr2)
        for a in preorder_traversal(dif):
            if isinstance(a, Float):
                dif = dif.xreplace({a: round(a, 15)})
        return _approx_sympy_expr(dif) == 0

    def __str__(self) -> str:
        sense = ">=" if self.sense == INEQ_GEQ_SENSE else "<="
        return f"{self.exprleft} {sense} {self.exprright}"

    def _list_unknown_vars(self, vars: Iterable[str]) -> List[VarAbs]:
        lvars_uknown = _list_unknown_vars(self.exprleft, vars)
        rvars_uknown = _list_unknown_vars(self.exprleft, vars)
        return list(lvars_uknown) + list(rvars_uknown)
