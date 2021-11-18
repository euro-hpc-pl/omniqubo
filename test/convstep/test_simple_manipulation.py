from copy import deepcopy

from omniqubo.convstep import MakeMax, MakeMin, RemoveConstraint
from omniqubo.sympyopt import SympyOpt
from omniqubo.sympyopt.constraints import ConstraintEq


class TestOneHot:
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
        sympyopt = conv.convert(sympyopt)
        assert sympyopt == sympyopt_max
        sympyopt = conv.convert(sympyopt)
        assert sympyopt == sympyopt_max

        conv = MakeMin()
        sympyopt = conv.convert(sympyopt)
        assert sympyopt == sympyopt_copy
        sympyopt = conv.convert(sympyopt)
        assert sympyopt == sympyopt_copy

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
        sympyopt = RemoveConstraint("second").convert(sympyopt)
        assert sympyopt_small == sympyopt
