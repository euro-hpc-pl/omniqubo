import re
from typing import Callable, Dict, List

from sympy import Expr, Symbol
from sympy.core.evalf import INF

from omniqubo.converters.converter import can_convert, convert
from omniqubo.converters.eq_to_objective import EqToObj
from omniqubo.converters.simple_manipulation import MakeMax, MakeMin, RemoveConstraint
from omniqubo.converters.utils import INTER_STR_SEP
from omniqubo.converters.varreplace import BitToSpin, TrivialIntToBit, VarOneHot, VarReplace
from omniqubo.models.sympyopt.constraints import ConstraintEq, ConstraintIneq
from omniqubo.models.sympyopt.vars import BitVar, IntVar

from .sympyopt import MAX_SENSE, MIN_SENSE, SympyOpt

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

    for cname in constr_names:
        c = model.constraints[cname]
        assert isinstance(c, ConstraintEq)
        if model.sense == MIN_SENSE:
            model.objective += converter.penalty * (c.exprleft - c.exprright) ** 2
        else:
            model.objective -= converter.penalty * (c.exprleft - c.exprright) ** 2
        model.constraints.pop(cname)
    return model


@can_convert.register
def can_convert_sympyopt_eqtoobj(model: SympyOpt, converter: EqToObj) -> bool:
    if converter.is_regexp:
        return True
    name = converter.name
    return name in model.constraints and type(model.constraints[name]) is ConstraintEq


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
    # TODO implement condition functions to data for interpret

    if converter.is_regexp:
        _rex = re.compile(converter.name)
        to_be_removed = []
        for cname in model.constraints:
            if _rex.fullmatch(cname):
                to_be_removed.append(cname)
        for cname in to_be_removed:
            model.constraints.pop(cname)
    else:
        model.constraints.pop(converter.name)

    return model


@can_convert.register
def can_convert_sympyopt_removeconstraint(model: SympyOpt, converter: RemoveConstraint) -> bool:
    if converter.is_regexp:
        return True
    return converter.name in model.constraints


# general commands for VarReplace


def _sub_expression(model: SympyOpt, rule_dict: Dict[Symbol, Expr]):
    model.objective = model.objective.xreplace(rule_dict)
    for c in model.constraints.values():
        if isinstance(c, (ConstraintEq, ConstraintIneq)):
            c.exprleft = c.exprleft.xreplace(rule_dict)
            c.exprright = c.exprright.xreplace(rule_dict)


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


# VarOneHot


def _get_expr_add_constr_onehot(model: SympyOpt, var: IntVar) -> Expr:
    name = var.name
    lb = var.lb
    ub = var.ub
    xs = [model.bit_var(f"{name}{INTER_STR_SEP}OH_{i}") for i in range(ub - lb + 1)]

    # add constraint
    c = ConstraintEq(sum(x for x in xs), 1)
    model.add_constraint(c, name=f"{INTER_STR_SEP}OH_{name}")

    return sum(v * x for x, v in zip(xs, range(lb, ub + 1)))


def _can_convert_onehot_sing(model: SympyOpt, name: str) -> bool:
    var = model.variables[name]
    if not isinstance(var, IntVar):
        return False
    return var.lb != -INF and var.ub != INF


@convert.register
def convert_sympyopt_varonehot(model: SympyOpt, converter: VarOneHot) -> SympyOpt:
    assert can_convert(model, converter)

    def filtering_fun(vname: str):
        return _can_convert_onehot_sing(model, vname)

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
    return _can_convert_onehot_sing(model, converter.varname)


# TrivialIntToBit:


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


def _get_expr_bittospin(model: SympyOpt, converter: BitToSpin, varname: str) -> Expr:
    var = model.spin_var(f"{varname}{INTER_STR_SEP}bts")
    if converter.reversed:
        return 1 - 2 * var
    else:
        return 1 + 2 * var


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
    print(rule_dict)
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
