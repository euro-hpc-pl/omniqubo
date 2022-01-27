from warnings import warn

import numpy as np
from multimethod import multimethod
from pandas import DataFrame
from sympy import Expr
from sympy.core.evalf import INF

from omniqubo.converters.converter import can_convert
from omniqubo.converters.eq_to_objective import EqToObj
from omniqubo.converters.simple_manipulation import MakeMax, MakeMin, RemoveConstraint
from omniqubo.converters.utils import INTER_STR_SEP
from omniqubo.converters.varreplace import BitToSpin, TrivialIntToBit, VarOneHot
from omniqubo.models.sympyopt.constraints import ConstraintEq, ConstraintIneq
from omniqubo.models.sympyopt.vars import BitVar, IntVar, VarAbsSympyOpt

from .sympyopt import MAX_SENSE, MIN_SENSE, SympyOpt

# EqToObj


@multimethod  # type: ignore[no-redef]
def convert(model: SympyOpt, converter: EqToObj):
    assert can_convert(model, converter)
    c = model.constraints[converter.name]
    assert isinstance(c, ConstraintEq)
    if model.sense == MIN_SENSE:
        model.objective += converter.penalty * (c.exprleft - c.exprright) ** 2
    else:
        model.objective -= converter.penalty * (c.exprleft - c.exprright) ** 2
    model.constraints.pop(converter.name)
    return model


@multimethod  # type: ignore[no-redef]
def can_convert(model: SympyOpt, converter: EqToObj) -> bool:
    name = converter.name
    return name in model.constraints and type(model.constraints[name]) == ConstraintEq


@multimethod  # type: ignore[no-redef]
def interpret(samples: DataFrame, converter: EqToObj) -> DataFrame:
    warn("EqToObj is not analysing feasibility")
    return samples


# MakeMax


@multimethod  # type: ignore[no-redef]
def convert(model: SympyOpt, converter: MakeMax) -> SympyOpt:
    assert can_convert(model, converter)
    if model.sense == MIN_SENSE:
        model.minimize(-model.get_objective())
        model.sense = MAX_SENSE
    return model


@multimethod  # type: ignore[no-redef]
def can_convert(model: SympyOpt, converter: MakeMax) -> bool:
    return True


@multimethod  # type: ignore[no-redef]
def interpret(samples: DataFrame, converter: MakeMax) -> DataFrame:
    return samples


# MakeMin


@multimethod  # type: ignore[no-redef]
def convert(model: SympyOpt, converter: MakeMin) -> SympyOpt:
    assert can_convert(model, converter)
    if model.sense == MAX_SENSE:
        model.minimize(-model.get_objective())
        model.sense = MIN_SENSE
    return model


@multimethod  # type: ignore[no-redef]
def can_convert(model: SympyOpt, converter: MakeMin) -> bool:
    return True


@multimethod  # type: ignore[no-redef]
def interpret(samples: DataFrame, converter: MakeMin) -> DataFrame:
    return samples


# RemoveConstraint


@multimethod  # type: ignore[no-redef]
def convert(model: SympyOpt, converter: RemoveConstraint) -> SympyOpt:
    assert can_convert(model, converter)
    model.constraints.pop(converter.name)
    return model


@multimethod  # type: ignore[no-redef]
def can_convert(model: SympyOpt, converter: RemoveConstraint) -> bool:
    return converter.name in model.constraints


@multimethod  # type: ignore[no-redef]
def interpret(samples: DataFrame, converter: RemoveConstraint) -> DataFrame:
    warn("Feasibility is not checked yet for RemoveConstraint")
    # TODO: add flag to RemoveConstraint
    return samples


# general commands for VarReplace


def _sub_expression(model: SympyOpt, expr: Expr, var: VarAbsSympyOpt):
    model.objective = model.objective.xreplace({var.var: expr})
    for c in model.constraints.values():
        if isinstance(c, (ConstraintEq, ConstraintIneq)):
            c.exprleft = c.exprleft.xreplace({var.var: expr})
            c.exprright = c.exprright.xreplace({var.var: expr})


# VarOneHot


def _get_expr_add_constr_onehot(model: SympyOpt, var: IntVar) -> Expr:
    name = var.name
    lb = var.lb
    ub = var.ub
    xs = [model.bit_var(f"{name}{INTER_STR_SEP}OH_{i}") for i in range(ub - lb + 1)]

    # add constraint
    c = ConstraintEq(sum(x for x in xs), 1)
    model.add_constraint(c, name=f"{INTER_STR_SEP}OH_{name}")

    # return
    return sum(v * x for x, v in zip(xs, range(lb, ub + 1)))


@multimethod  # type: ignore[no-redef]
def convert(model: SympyOpt, converter: VarOneHot) -> SympyOpt:
    assert can_convert(model, converter)
    var = model.variables[converter.varname]
    assert isinstance(var, IntVar)

    sub_expr = _get_expr_add_constr_onehot(model, var)

    # substitute expression
    _sub_expression(model, sub_expr, var)

    model.variables.pop(converter.varname)
    converter.data["lb"] = var.lb
    converter.data["ub"] = var.ub
    return model


@multimethod  # type: ignore[no-redef]
def can_convert(model: SympyOpt, converter: VarOneHot) -> bool:
    var = model.variables[converter.varname]
    if not isinstance(var, IntVar):
        return False
    return var.lb != -INF and var.ub != INF


@multimethod  # type: ignore[no-redef]
def interpret(samples: DataFrame, converter: VarOneHot) -> DataFrame:
    name = converter.varname
    lb = converter.data["lb"]
    ub = converter.data["ub"]

    names = [f"{name}{INTER_STR_SEP}OH_{i}" for i in range(ub - lb + 1)]
    samples["feasible_tmp"] = samples.apply(lambda row: sum(row[n] for n in names) == 1, axis=1)

    def set_var_value(row):
        return lb + [row[n] for n in names].index(1) if row["feasible_tmp"] else np.nan

    samples[name] = samples.apply(
        set_var_value,
        axis=1,
    )

    samples["feasible"] &= samples.pop("feasible_tmp")
    for n in names:
        samples.pop(n)

    return samples


# TrivialIntToBit:


@multimethod  # type: ignore[no-redef]
def convert(model: SympyOpt, converter: TrivialIntToBit) -> SympyOpt:
    assert can_convert(model, converter)
    var = model.variables[converter.varname]
    assert isinstance(var, IntVar)

    bit_var = model.bit_var(f"{converter.varname}{INTER_STR_SEP}itb")
    sub_expr = var.lb + bit_var

    # substitute expression
    _sub_expression(model, sub_expr, var)

    model.variables.pop(converter.varname)
    converter.data["lb"] = var.lb
    return model


@multimethod  # type: ignore[no-redef]
def can_convert(model: SympyOpt, converter: TrivialIntToBit) -> bool:
    var = model.variables[converter.varname]
    return isinstance(var, IntVar) and var.ub - var.lb == 1


@multimethod  # type: ignore[no-redef]
def interpret(samples: DataFrame, converter: TrivialIntToBit) -> DataFrame:
    name_orig = converter.varname
    name_new = f"{name_orig}{INTER_STR_SEP}itb"
    samples.rename(columns={name_new: name_orig})
    samples[name_orig] = samples[name_orig] + converter.data["lb"]
    return samples


#  BitToSpin


def _get_expr_add_constr_bittospin(model: SympyOpt, converter: BitToSpin) -> Expr:
    var = model.spin_var(f"{converter.varname}{INTER_STR_SEP}bts")
    if converter.reversed:
        return (1 - var) / 2
    else:
        return (1 + var) / 2


@multimethod  # type: ignore[no-redef]
def convert(model: SympyOpt, converter: BitToSpin) -> SympyOpt:
    assert can_convert(model, converter)
    var = model.variables[converter.varname]
    assert isinstance(var, BitVar)

    sub_expr = _get_expr_add_constr_bittospin(model, converter)

    # substitute expression
    _sub_expression(model, sub_expr, var)

    model.variables.pop(converter.varname)
    return model


@multimethod  # type: ignore[no-redef]
def can_convert(model: SympyOpt, converter: BitToSpin) -> bool:
    return isinstance(model.variables[converter.varname], BitVar)


@multimethod  # type: ignore[no-redef]
def interpret(samples: DataFrame, converter: BitToSpin) -> DataFrame:
    raise NotImplementedError()
