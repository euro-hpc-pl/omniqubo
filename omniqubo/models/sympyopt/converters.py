import re
from math import ceil, prod
from typing import Callable, Dict, List, Union

from pandas import DataFrame
from sympy import Add, Expr, Integer, Mul, Number, Pow, Symbol, expand, lambdify
from sympy.core.evalf import INF

from omniqubo.converters.converter import can_convert, convert
from omniqubo.converters.eq_to_objective import EqToObj
from omniqubo.converters.ineq_to_eq import IneqToEq
from omniqubo.converters.simple_manipulation import MakeMax, MakeMin, RemoveConstraint
from omniqubo.converters.utils import INTER_STR_SEP
from omniqubo.converters.varreplace import (
    BitToSpin,
    TrivialIntToBit,
    VarBinary,
    VarOneHot,
    VarReplace,
    _binary_encoding_coeff,
)
from omniqubo.models.sympyopt.constraints import INEQ_GEQ_SENSE, ConstraintEq, ConstraintIneq
from omniqubo.models.sympyopt.vars import BitVar, IntVar

from .sympyopt import MAX_SENSE, MIN_SENSE, SympyOpt

# for explanation of how each convert and can_convert works, see documentation
# of appropriate converter class


# transforms sympy-based constraint a ==/<=/>= b into a function computing a - b
# https://stackoverflow.com/questions/64475745/how-to-express-dataframe-operations-using-symbols
def _eq_to_verifier(c: Union[ConstraintEq, ConstraintIneq]):
    def verifier(df: DataFrame):
        expr = c.exprleft - c.exprright
        callable_obj = lambdify(list(expr.free_symbols), expr)
        return callable_obj(**{str(a): df[str(a)] for a in expr.atoms() if isinstance(a, Symbol)})

    return verifier


# EqToObj


@convert.register
def convert_sympyopt_eqtoobj(model: SympyOpt, converter: EqToObj):
    assert can_convert(model, converter)
    constr_names: List[str] = []
    if converter.is_regexp:
        _rex = re.compile(converter.name)
        for cname in model.constraints:
            c = model.constraints[cname]
            if _rex.fullmatch(cname) and isinstance(c, ConstraintEq):
                constr_names.append(cname)
    else:
        assert isinstance(model.constraints[converter.name], ConstraintEq)
        constr_names.append(converter.name)

    converter.data["verifiers"] = []
    for cname in constr_names:
        c = model.constraints.pop(cname)
        assert isinstance(c, ConstraintEq)
        if model.sense == MIN_SENSE:
            model.objective += converter.penalty * (c.exprleft - c.exprright) ** 2
        else:
            model.objective -= converter.penalty * (c.exprleft - c.exprright) ** 2
        if c.check_interpret:
            converter.data["verifiers"].append(_eq_to_verifier(c))
    return model


@can_convert.register
def can_convert_sympyopt_eqtoobj(model: SympyOpt, converter: EqToObj) -> bool:
    if converter.is_regexp:
        return True
    name = converter.name
    if name not in model.constraints:
        return False
    return isinstance(model.constraints[name], ConstraintEq)


# IneqToEq

# if lbs and ubs are bound of expressions in a product, outputs the smallest
# possible value
# it assumes that list1 is element-wise smaller than list2 of the same size
def _get_min_product(lbs: List[float], ubs: List[float]) -> float:
    switchers_number = sum(lb * ub <= 0 for lb, ub in zip(lbs, ubs))
    if switchers_number == 0:
        # sign will be always the same irrespectively of chosen lb or ub
        neg_no = sum(lb < 0 for lb in lbs)
        if neg_no % 2 == 0:
            return prod(min(abs(lb), abs(ub)) for lb, ub in zip(lbs, ubs))
        else:
            return -prod(max(abs(lb), abs(ub)) for lb, ub in zip(lbs, ubs))
    else:
        # here we can always choose lb or ub to be negative
        result = prod(ub if abs(lb) < abs(ub) else lb for lb, ub in zip(lbs, ubs))
        if result <= 0:
            return result
        # we need to switch one variable
        values = []
        for lb, ub in zip(lbs, ubs):
            chosen, not_chosen = (ub, lb) if abs(lb) < abs(ub) else (lb, ub)
            values.append(result * not_chosen / chosen)
        return min(values)


# if lbs and ubs are bound of expressions in a product, outputs the largest
# possible value
# it assumes that list1 is element-wise smaller than list2 of the same size
def _get_max_product(lbs: List[float], ubs: List[float]) -> float:
    return -_get_min_product([-x for x in ubs], [-x for x in lbs])


# gets upperbound on the sympy expression. Model is used for getting var bounds
# tight for linear expression
def _get_upperbound(expr: Expr, model: SympyOpt) -> float:
    if isinstance(expr, Symbol):
        return model.variables[expr.name].get_ub()
    elif isinstance(expr, Number):
        return float(expr.evalf())
    elif isinstance(expr, Add):
        return sum(_get_upperbound(x, model) for x in expr._args)
    elif isinstance(expr, Mul):
        # this is complicated as it depends on the number of negative values
        # chosen
        lbs = [_get_lowerbound(x, model) for x in expr._args]
        ubs = [_get_upperbound(x, model) for x in expr._args]

        return _get_max_product(lbs, ubs)
    elif isinstance(expr, Pow):
        # HACK 1: it assumes exponent is an integer
        # HACK 2: does not work tightly with spins. Would require a lowerbound
        # with abs, which is difficult to implement
        assert isinstance(expr.exp, Integer)
        baselb = _get_lowerbound(expr.base, model)
        baseub = _get_upperbound(expr.base, model)
        return max(float(baselb ** expr.exp), float(baseub ** expr.exp))
    else:
        raise ValueError(f"Algebraic expression {type(expr)} cannot be handled")


# gets lowerbound on the sympy expression. Model is used for getting var bounds
# tight for linear expression
def _get_lowerbound(expr: Expr, model: SympyOpt) -> float:
    if isinstance(expr, Symbol):
        return model.variables[expr.name].get_lb()
    elif isinstance(expr, Number):
        return float(expr.evalf())
    elif isinstance(expr, Add):
        return sum(_get_lowerbound(x, model) for x in expr._args)
    elif isinstance(expr, Mul):
        # this is complicated as it depends on the number of negative values
        # chosen
        lbs = [_get_lowerbound(x, model) for x in expr._args]
        ubs = [_get_upperbound(x, model) for x in expr._args]

        return _get_min_product(lbs, ubs)
    elif isinstance(expr, Pow):
        # HACK 1: it assumes exponent is an integer
        # HACK 2: does not work tightly with spins. Would require a lowerbound
        # with abs, which is difficult to implement
        assert isinstance(expr.exp, Integer)
        if expr.exp % 2 == 1:
            return float(_get_lowerbound(expr.base, model) ** expr.exp)
        else:
            baselb = _get_lowerbound(expr.base, model)
            baseub = _get_upperbound(expr.base, model)
            if baselb * baseub <= 0:
                return 0
            else:
                return min(float(baselb ** expr.exp), float(baseub ** expr.exp))
    else:
        raise ValueError(f"Algebraic expression {type(expr)} cannot be handled")


@convert.register
def convert_sympyopt_ineqtoeq(model: SympyOpt, converter: IneqToEq):
    assert can_convert(model, converter)

    constr_names: List[str] = []
    if converter.is_regexp:
        _rex = re.compile(converter.name)
        for cname in model.constraints:
            c = model.constraints[cname]
            if _rex.fullmatch(cname) and isinstance(c, ConstraintIneq):
                constr_names.append(cname)
    else:
        assert isinstance(model.constraints[converter.name], ConstraintIneq)
        constr_names.append(converter.name)

    converter.data["verifiers"] = []
    for cname in constr_names:
        c = model.constraints.pop(cname)
        assert isinstance(c, ConstraintIneq)

        if c.sense == INEQ_GEQ_SENSE:
            slack_bound = -_get_lowerbound(expand(c.exprright - c.exprleft), model)
        else:  # INEQ_LEQ_SENSE
            slack_bound = -_get_lowerbound(expand(c.exprleft - c.exprright), model)
        slack_name = f"{cname}{INTER_STR_SEP}slack"
        if slack_bound == 0:
            slack_var = 0  # no need for a variable
            slack_name = ""
        elif slack_bound > 0:
            slack_var = model.int_var(slack_name, lb=0, ub=ceil(slack_bound))
        else:
            raise ValueError(f"Inequality {cname} is not satisfiable")

        if c.sense == INEQ_GEQ_SENSE:
            c_new = ConstraintEq(c.exprleft - slack_var, c.exprright)
        else:  # INEQ_LEQ_SENSE
            c_new = ConstraintEq(c.exprleft + slack_var, c.exprright)
        c_new.check_interpret = False
        model.add_constraint(c_new, cname)

        if converter.check_slack:
            converter.data["verifiers"].append((_eq_to_verifier(c_new), slack_name))
        else:
            ctype = "geq" if c.sense == INEQ_GEQ_SENSE else "leq"
            converter.data["verifiers"].append((_eq_to_verifier(c), slack_name, ctype))

    return model


@can_convert.register
def can_convert_sympyopt_ineqtoeq(model: SympyOpt, converter: IneqToEq) -> bool:
    if converter.is_regexp:
        return True
    name = converter.name
    if name not in model.constraints:
        return False
    return isinstance(model.constraints[name], ConstraintIneq)


# MakeMax


@convert.register
def convert_sympyopt_makemax(model: SympyOpt, converter: MakeMax) -> SympyOpt:
    assert can_convert(model, converter)
    if model.sense == MIN_SENSE:
        model.minimize(-model.get_objective())
        model.sense = MAX_SENSE
    return model


@can_convert.register
def can_convert_sympyopt_makemax(model: SympyOpt, converter: MakeMax) -> bool:
    return True


# MakeMin


@convert.register
def convert_sympyopt_makemin(model: SympyOpt, converter: MakeMin) -> SympyOpt:
    assert can_convert(model, converter)
    if model.sense == MAX_SENSE:
        model.minimize(-model.get_objective())
        model.sense = MIN_SENSE
    return model


@can_convert.register
def can_convert_sympyopt_makemin(model: SympyOpt, converter: MakeMin) -> bool:
    return True


# RemoveConstraint


@convert.register
def convert_sympyopt_removeconstraint(model: SympyOpt, converter: RemoveConstraint) -> SympyOpt:
    assert can_convert(model, converter)
    converter.data["verifiers"] = []

    to_be_removed: List[str] = []
    if converter.is_regexp:
        _rex = re.compile(converter.name)
        to_be_removed = []
        for cname in model.constraints:
            if _rex.fullmatch(cname):
                to_be_removed.append(cname)
    else:
        to_be_removed.append(converter.name)

    for cname in to_be_removed:
        c = model.constraints.pop(cname)

        if converter.check_constraint:
            if isinstance(c, ConstraintEq):
                ctype = "eq"
            else:
                assert isinstance(c, ConstraintIneq)
                ctype = "geq" if c.sense == INEQ_GEQ_SENSE else "leq"
            assert isinstance(c, (ConstraintEq, ConstraintIneq))
            converter.data["verifiers"].append((_eq_to_verifier(c), ctype))
    return model


@can_convert.register
def can_convert_sympyopt_removeconstraint(model: SympyOpt, converter: RemoveConstraint) -> bool:
    if converter.is_regexp:
        return True
    return converter.name in model.constraints


# general commands for VarReplace

# substitute expression for symbols for objective and all constraints
# note: rule_dict is much faster than replacing symbols one by one
def _sub_expression(model: SympyOpt, rule_dict: Dict[Symbol, Expr]):
    model.objective = model.objective.xreplace(rule_dict)
    for c in model.constraints.values():
        if isinstance(c, (ConstraintEq, ConstraintIneq)):
            c.exprleft = c.exprleft.xreplace(rule_dict)
            c.exprright = c.exprright.xreplace(rule_dict)


# looks for a matching variables names according to the name (perhaps regular
# expression). if is regular expression - filter the varnames. Otherwise
# filtering_fun has to be satisfied
def _matching_varnames(model: SympyOpt, converter: VarReplace, filtering_fun: Callable):
    var_to_replace: List[str] = []
    if converter.is_regexp:
        _rex = re.compile(converter.varname)
        for varname in model.variables:
            if _rex.fullmatch(varname) and filtering_fun(varname):
                var_to_replace.append(varname)
    else:
        assert filtering_fun(converter.varname)
        var_to_replace.append(converter.varname)
    return var_to_replace


# check if integer variable can converted according to the One-hot encoding
def _can_convert_int(model: SympyOpt, name: str) -> bool:
    var = model.variables[name]
    if not isinstance(var, IntVar):
        return False
    return var.lb != -INF and var.ub != INF


# VarOneHot

# outputs expression and adds constraint for one-hot encoding
def _get_expr_add_constr_onehot(model: SympyOpt, var: IntVar) -> Expr:
    name = var.name
    lb = var.lb
    ub = var.ub
    xs = [model.bit_var(f"{name}{INTER_STR_SEP}OH_{i}") for i in range(ub - lb + 1)]

    # add constraint
    c = ConstraintEq(sum(x for x in xs), 1)
    model.add_constraint(c, name=f"{INTER_STR_SEP}OH_{name}")

    return sum(v * x for x, v in zip(xs, range(lb, ub + 1)))


@convert.register
def convert_sympyopt_varonehot(model: SympyOpt, converter: VarOneHot) -> SympyOpt:
    assert can_convert(model, converter)

    def filtering_fun(vname: str):
        return _can_convert_int(model, vname)

    var_to_replace = _matching_varnames(model, converter, filtering_fun)

    rule_dict = dict()
    for vname in var_to_replace:
        var = model.variables[vname]
        assert isinstance(var, IntVar)
        rule_dict[var.var] = _get_expr_add_constr_onehot(model, var)

    _sub_expression(model, rule_dict)

    converter.data["bounds"] = dict()
    for vname in var_to_replace:
        var = model.variables.pop(vname)
        assert isinstance(var, IntVar)
        converter.data["bounds"][vname] = (var.lb, var.ub)
    return model


@can_convert.register
def can_convert_sympyopt_varonehot(model: SympyOpt, converter: VarOneHot) -> bool:
    if converter.is_regexp:
        return True
    return _can_convert_int(model, converter.varname)


# VarBinary

# https://link.springer.com/article/10.1007/s11128-019-2213-x Eq. (5)
def _get_expr_binary(model: SympyOpt, var: IntVar) -> Expr:
    name = var.name
    lb: int = var.lb
    ub: int = var.ub
    vals = _binary_encoding_coeff(lb, ub)
    vars = [model.bit_var(f"{name}{INTER_STR_SEP}BIN_{i}") for i in range(len(vals))]
    return lb + sum(val * x for val, x in zip(vals, vars))


@convert.register
def convert_sympyopt_varbinary(model: SympyOpt, converter: VarBinary) -> SympyOpt:
    assert can_convert(model, converter)

    def filtering_fun(vname: str):
        return _can_convert_int(model, vname)

    var_to_replace = _matching_varnames(model, converter, filtering_fun)

    rule_dict = dict()
    for vname in var_to_replace:
        var = model.variables[vname]
        assert isinstance(var, IntVar)
        rule_dict[var.var] = _get_expr_binary(model, var)

    _sub_expression(model, rule_dict)

    converter.data["bounds"] = dict()
    for vname in var_to_replace:
        var = model.variables.pop(vname)
        assert isinstance(var, IntVar)
        converter.data["bounds"][vname] = (var.lb, var.ub)
    return model


@can_convert.register
def can_convert_sympyopt_varbinary(model: SympyOpt, converter: VarOneHot) -> bool:
    if converter.is_regexp:
        return True
    return _can_convert_int(model, converter.varname)


# TrivialIntToBit:

# checks if variables can be converted according to TrivialIntToBit (lb <= y <=
# lb+1)
def _can_convert_trivitb_sing(model: SympyOpt, name: str) -> bool:
    var = model.variables[name]
    return isinstance(var, IntVar) and var.ub - var.lb == 1


@convert.register
def convert_sympyopt_trivialinttobit(model: SympyOpt, converter: TrivialIntToBit) -> SympyOpt:
    assert can_convert(model, converter)
    if converter.is_regexp:

        def filtering_fun(vname: str):
            return _can_convert_trivitb_sing(model, vname)

    else:

        def filtering_fun(vname: str):
            return True

    var_to_replace = _matching_varnames(model, converter, filtering_fun)
    # if not converter.is_regexp, check if we can implement it and do nothing if
    # you cannot
    if not converter.is_regexp:
        if not _can_convert_trivitb_sing(model, var_to_replace[0]):
            return model

    rule_dict = dict()
    for vname in var_to_replace:
        var = model.variables[vname]
        assert isinstance(var, IntVar)  # for mypy
        bit_var = model.bit_var(f"{vname}{INTER_STR_SEP}itb")
        rule_dict[var.var] = var.lb + bit_var

    _sub_expression(model, rule_dict)

    converter.data["lb"] = dict()
    for vname in var_to_replace:
        var = model.variables.pop(vname)
        assert isinstance(var, IntVar)
        converter.data["lb"][vname] = var.lb
    return model


@can_convert.register
def can_convert_sympyopt_trivialinttobit(model: SympyOpt, converter: TrivialIntToBit) -> bool:
    return True  # always can convert, even if does nothing


#  BitToSpin

# outputs expression transforming Bit to Spin. Note two expressions are
# possible, and the reversed one is more popular in the literature
def _get_expr_bittospin(model: SympyOpt, converter: BitToSpin, varname: str) -> Expr:
    var = model.spin_var(f"{varname}{INTER_STR_SEP}bts")
    if converter.reversed:
        return 1 - 2 * var
    else:
        return 1 + 2 * var


# checks if variable is binary and thus can be converted to spin
def _can_convert_bittospin_sing(model: SympyOpt, name: str) -> bool:
    return isinstance(model.variables[name], BitVar)


@convert.register
def convert_sympyopt_bittospin(model: SympyOpt, converter: BitToSpin) -> SympyOpt:
    assert can_convert(model, converter)

    def filtering_fun(vname: str):
        return _can_convert_bittospin_sing(model, vname)

    var_to_replace = _matching_varnames(model, converter, filtering_fun)

    rule_dict = dict()
    for vname in var_to_replace:
        var = model.variables[vname].var
        rule_dict[var] = _get_expr_bittospin(model, converter, vname)
    _sub_expression(model, rule_dict)

    converter.data["varnames"] = set(var_to_replace)
    for vname in var_to_replace:
        model.variables.pop(vname)

    return model


@can_convert.register
def can_convert_sympyopt_bittospin(model: SympyOpt, converter: BitToSpin) -> bool:
    if converter.is_regexp:
        return True
    return _can_convert_bittospin_sing(model, converter.varname)
