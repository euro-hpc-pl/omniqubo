from copy import deepcopy


def _list_unknown_vars(obj, vars):
    return filter(lambda v: v not in vars, obj.free_symbols)


class ConstraintAbs:
    def __init__(self) -> None:
        self.exprleft = None
        self.exprright = None
        pass

    def is_eq_constraint(self):
        return False

    def is_ineq_constraint(self):
        return False

    def is_type_constraint(self):
        return False

    def _list_unknown_vars(self, vars):
        lvars_uknown = _list_unknown_vars(self.exprleft, vars)
        rvars_uknown = _list_unknown_vars(self.exprleft, vars)
        return list(lvars_uknown) + list(rvars_uknown)


class ConstraintEq(ConstraintAbs):
    def __init__(self, exprleft, exprright):
        self.exprleft = deepcopy(exprleft)
        self.exprright = deepcopy(exprright)

    def is_eq_constraint(self):
        True


INEQ_LEQ_SENSE = "leq"
INEQ_GEQ_SENSE = "geq"


class ConstraintIneq(ConstraintAbs):
    def __init__(self, exprleft, exprright, sense):
        self.exprleft = deepcopy(exprleft)
        self.exprright = deepcopy(exprright)
        if sense != INEQ_GEQ_SENSE and sense != INEQ_LEQ_SENSE:
            raise ValueError(f"incorrect sense {sense}")
        self.sense = sense

    def is_ineq_constraint(self):
        True
