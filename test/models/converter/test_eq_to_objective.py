import warnings

import pytest

from omniqubo.converters.eq_to_objective import EqToObj
from omniqubo.models.sympyopt.constraints import ConstraintEq
from omniqubo.models.sympyopt.converters import convert
from omniqubo.models.sympyopt.sympyopt import SympyOpt


class TestEqToObj:
    def test_zero_penalty_warning(self):
        with pytest.warns(Warning):
            EqToObj(".*", True, 0.0)

        with warnings.catch_warnings():
            EqToObj(".*", True, 1.0)

    def test_linear_min(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.minimize(2 * x - 3 * y + 2)
        sympyopt.add_constraint(ConstraintEq(2 * x - 3 * y, 3), name="constr1")
        sympyopt.add_constraint(ConstraintEq(2 * x ** 2 - 3 * y, 0), name="constr2")
        conv1 = EqToObj("constr1", False, 10)
        conv2 = EqToObj("constr2", False, 3.5)
        sympyopt = convert(sympyopt, conv1)
        sympyopt = convert(sympyopt, conv2)

        sympyopt2 = SympyOpt()
        x = sympyopt2.int_var(name="x", lb=0, ub=2)
        y = sympyopt2.int_var(lb=-2, ub=3, name="y")
        sympyopt2.minimize(
            2 * x - 3 * y + 2 + 10 * (2 * x - 3 * y - 3) ** 2 + 3.5 * (2 * x ** 2 - 3 * y) ** 2
        )
        assert sympyopt2 == sympyopt

    def test_linear_max(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.maximize(2 * x - 3 * y + 2)
        sympyopt.add_constraint(ConstraintEq(2 * x - 3 * y, 3), name="constr1")
        sympyopt.add_constraint(ConstraintEq(2 * x ** 2 - 3 * y, 0), name="constr2")
        conv1 = EqToObj("constr1", False, 10)
        conv2 = EqToObj("constr2", False, 3.5)
        sympyopt = convert(sympyopt, conv1)
        sympyopt = convert(sympyopt, conv2)

        sympyopt2 = SympyOpt()
        x = sympyopt2.int_var(name="x", lb=0, ub=2)
        y = sympyopt2.int_var(lb=-2, ub=3, name="y")
        sympyopt2.maximize(
            2 * x - 3 * y + 2 - 10 * (2 * x - 3 * y - 3) ** 2 - 3.5 * (2 * x ** 2 - 3 * y) ** 2
        )
        assert sympyopt2 == sympyopt
