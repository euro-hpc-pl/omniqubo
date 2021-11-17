from omniqubo.convstep import EqToObj
from omniqubo.sympyopt import SympyOpt
from omniqubo.sympyopt.constraints import ConstraintEq


class TestEqToObj:
    def test_linear_min(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.minimize(2 * x - 3 * y + 2)
        sympyopt.add_constraint(ConstraintEq(2 * x - 3 * y, 3), name="constr1")
        sympyopt.add_constraint(ConstraintEq(2 * x ** 2 - 3 * y, 0), name="constr2")
        conv1 = EqToObj("constr1", 10)
        conv2 = EqToObj("constr2", 3.5)
        sympyopt = conv1.convert(sympyopt)
        sympyopt = conv2.convert(sympyopt)

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
        conv1 = EqToObj("constr1", 10)
        conv2 = EqToObj("constr2", 3.5)
        sympyopt = conv1.convert(sympyopt)
        sympyopt = conv2.convert(sympyopt)

        sympyopt2 = SympyOpt()
        x = sympyopt2.int_var(name="x", lb=0, ub=2)
        y = sympyopt2.int_var(lb=-2, ub=3, name="y")
        sympyopt2.maximize(
            2 * x - 3 * y + 2 - 10 * (2 * x - 3 * y - 3) ** 2 - 3.5 * (2 * x ** 2 - 3 * y) ** 2
        )
        assert sympyopt2 == sympyopt
