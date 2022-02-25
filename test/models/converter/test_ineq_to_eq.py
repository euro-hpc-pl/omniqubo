from copy import deepcopy

import pytest
from pandas import DataFrame

from omniqubo.converters.converter import interpret
from omniqubo.converters.ineq_to_eq import IneqToEq
from omniqubo.models.sympyopt.constraints import (
    INEQ_GEQ_SENSE,
    INEQ_LEQ_SENSE,
    ConstraintEq,
    ConstraintIneq,
)
from omniqubo.models.sympyopt.converters import convert
from omniqubo.models.sympyopt.sympyopt import SympyOpt


class TestIneqToEq:
    def test_convert(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.minimize(2 * x - 3 * y + 2)
        sympyopt.add_constraint(ConstraintIneq(2 * x - 3 * y, 3, INEQ_GEQ_SENSE), name="constr1")
        sympyopt.add_constraint(
            ConstraintIneq(2 * x ** 2 - 3 * y, 0, INEQ_LEQ_SENSE), name="constr2"
        )
        sympyopt.add_constraint(
            ConstraintIneq(2 * x ** 2 - 3 * y ** 3, 0, INEQ_LEQ_SENSE), name="constr3"
        )
        conv1 = IneqToEq("constr1", False, check_slack=False)
        conv2 = IneqToEq("constr2", False, check_slack=False)
        conv3 = IneqToEq("constr3", False, check_slack=False)
        sympyopt = convert(sympyopt, conv1)
        sympyopt = convert(sympyopt, conv2)
        sympyopt = convert(sympyopt, conv3)

        assert sympyopt.variables["constr1___slack"].get_lb() == 0
        assert sympyopt.variables["constr1___slack"].get_ub() == 7
        assert sympyopt.variables["constr2___slack"].get_lb() == 0
        assert sympyopt.variables["constr2___slack"].get_ub() == 9
        assert sympyopt.variables["constr3___slack"].get_lb() == 0
        assert sympyopt.variables["constr3___slack"].get_ub() == 81

        sympyopt3 = deepcopy(sympyopt)
        sympyopt3 = convert(sympyopt, IneqToEq("constr1", True, check_slack=False))
        assert sympyopt3 == sympyopt

        sympyopt2 = SympyOpt()
        x = sympyopt2.int_var(name="x", lb=0, ub=2)
        y = sympyopt2.int_var(lb=-2, ub=3, name="y")
        xi1 = sympyopt2.int_var(lb=0, ub=7, name="constr1___slack")
        xi2 = sympyopt2.int_var(lb=0, ub=9, name="constr2___slack")
        xi3 = sympyopt2.int_var(lb=0, ub=81, name="constr3___slack")
        sympyopt2.minimize(2 * x - 3 * y + 2)
        sympyopt2.add_constraint(ConstraintEq(2 * x - 3 * y - xi1, 3), name="constr1")
        sympyopt2.add_constraint(ConstraintEq(2 * x ** 2 - 3 * y + xi2, 0), name="constr2")
        sympyopt2.add_constraint(ConstraintEq(2 * x ** 2 - 3 * y ** 3 + xi3, 0), name="constr3")
        assert sympyopt2 == sympyopt

    def test_zero_slack_geq(self):
        # geq
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.add_constraint(ConstraintIneq(2 * x - 3 * y, 10, INEQ_GEQ_SENSE), name="c1")
        conv = IneqToEq(".*", True, check_slack=False)
        sympyopt = convert(sympyopt, conv)

        sympyopt2 = SympyOpt()
        x = sympyopt2.int_var(name="x", lb=0, ub=2)
        y = sympyopt2.int_var(lb=-2, ub=3, name="y")
        sympyopt2.add_constraint(ConstraintEq(2 * x - 3 * y, 10), name="c1")
        assert sympyopt2 == sympyopt

        samples1 = DataFrame({"x": [2], "y": [-2], "feasible": [True]})
        samples2 = DataFrame({"x": [1], "y": [0], "feasible": [True]})
        assert interpret(samples1, conv)["feasible"][0]
        assert not interpret(samples2, conv)["feasible"][0]

        # leq
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.add_constraint(ConstraintIneq(2 * x + y, -2, INEQ_LEQ_SENSE), name="c2")
        conv = IneqToEq(".*", True, check_slack=False)
        sympyopt = convert(sympyopt, conv)

        sympyopt2 = SympyOpt()
        x = sympyopt2.int_var(name="x", lb=0, ub=2)
        y = sympyopt2.int_var(lb=-2, ub=3, name="y")
        sympyopt2.add_constraint(ConstraintEq(2 * x + y, -2), name="c2")
        assert sympyopt2 == sympyopt

        samples1 = DataFrame({"x": [0], "y": [-2], "feasible": [True]})
        samples2 = DataFrame({"x": [1], "y": [3], "feasible": [True]})
        assert interpret(samples1, conv)["feasible"][0]
        assert not interpret(samples2, conv)["feasible"][0]

    def test_infeasible(self):
        # geq
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.add_constraint(ConstraintIneq(2 * x - 3 * y, 11, INEQ_GEQ_SENSE), name="c1")
        conv = IneqToEq(".*", True, check_slack=False)
        with pytest.raises(ValueError):
            sympyopt = convert(sympyopt, conv)

        # leq
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.add_constraint(ConstraintIneq(2 * x + y, -3, INEQ_LEQ_SENSE), name="c2")
        conv = IneqToEq(".*", True, check_slack=False)
        with pytest.raises(ValueError):
            sympyopt = convert(sympyopt, conv)
