from copy import deepcopy

import numpy as np
import pytest
from pandas import DataFrame

from omniqubo.converters.converter import interpret
from omniqubo.converters.simple_manipulation import (
    MakeMax,
    MakeMin,
    RemoveConstraint,
    SetIntVarBounds,
)
from omniqubo.models.sympyopt.constraints import ConstraintEq
from omniqubo.models.sympyopt.converters import convert
from omniqubo.models.sympyopt.sympyopt import SympyOpt


class TestSimpleManipulation:
    def test_objective_minmax(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.minimize(2 * x - 3 * y + 2)
        sympyopt_copy = deepcopy(sympyopt)

        sympyopt_max = SympyOpt()
        x = sympyopt_max.int_var(name="x", lb=0, ub=2)
        y = sympyopt_max.int_var(lb=-2, ub=3, name="y")
        sympyopt_max.maximize(-2 * x + 3 * y - 2)
        assert sympyopt != sympyopt_max

        conv = MakeMax()
        sympyopt = convert(sympyopt, conv)
        assert sympyopt == sympyopt_max
        sympyopt = convert(sympyopt, conv)
        assert sympyopt == sympyopt_max

        conv = MakeMin()
        sympyopt = convert(sympyopt, conv)
        assert sympyopt == sympyopt_copy
        sympyopt = convert(sympyopt, conv)
        assert sympyopt == sympyopt_copy

    def test_makeminmax_interpret(self):
        # makemin/makemax should not change anything thus it can be tested on
        # random dataframes
        df = DataFrame(np.random.randint(0, 100, size=(10, 4)), columns=list("ABCD"))
        df_new = interpret(df, MakeMax())
        assert df.equals(df_new)
        df_new = interpret(df, MakeMin())
        assert df.equals(df_new)

    def test_remove_constraint(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.minimize(2 * x - 3 * y + 2)
        c = ConstraintEq(2 * x + 3 * y, -2 - y ** 2)
        sympyopt.add_constraint(c, "first")
        sympyopt_small = deepcopy(sympyopt)

        c = ConstraintEq(2 * x + 3 * y, -2 - y)
        sympyopt.add_constraint(c, "second")

        assert sympyopt_small != sympyopt
        sympyopt = convert(sympyopt, RemoveConstraint("second", False, False))
        assert sympyopt_small == sympyopt

    def test_int_bounds(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x")
        y = sympyopt.int_var(name="y", ub=3)
        sympyopt.minimize(2 * x - 3 * y + 2)
        sympyopt.add_constraint(ConstraintEq(2 * x + 3 * y, -2 - y ** 2), "first")
        with pytest.raises(AssertionError):
            sympyopt = convert(sympyopt, SetIntVarBounds("y", False, 5, None))
        sympyopt = convert(sympyopt, SetIntVarBounds("y", False, -2, None))

        sympyopt2 = SympyOpt()
        x = sympyopt2.int_var(name="x")
        y = sympyopt2.int_var(name="y", lb=-2, ub=3)
        sympyopt2.minimize(2 * x - 3 * y + 2)
        sympyopt2.add_constraint(ConstraintEq(2 * x + 3 * y, -2 - y ** 2), "first")
        assert sympyopt2 == sympyopt

        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x")
        y = sympyopt.int_var(name="y", ub=3)
        sympyopt.minimize(2 * x - 3 * y + 2)
        sympyopt.add_constraint(ConstraintEq(2 * x + 3 * y, -2 - y ** 2), "first")
        sympyopt = convert(sympyopt, SetIntVarBounds(".*", True, 0, 1))

        sympyopt2 = SympyOpt()
        x = sympyopt2.int_var(name="x", lb=0, ub=1)
        y = sympyopt2.int_var(name="y", lb=0, ub=3)
        sympyopt2.minimize(2 * x - 3 * y + 2)
        sympyopt2.add_constraint(ConstraintEq(2 * x + 3 * y, -2 - y ** 2), "first")
        print(sympyopt)
        print(sympyopt2)
        assert sympyopt2 == sympyopt
