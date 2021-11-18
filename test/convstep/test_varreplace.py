from omniqubo.convstep import VarOneHot
from omniqubo.sympyopt import SympyOpt
from omniqubo.sympyopt.constraints import ConstraintEq


class TestOneHot:
    def test_objective(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.minimize(2 * x - 3 * y + 2)
        conv = VarOneHot(sympyopt.variables["y"])
        sympyopt = conv.convert(sympyopt)

        sympyopt2 = SympyOpt()
        x = sympyopt2.int_var(name="x", lb=0, ub=2)
        y1 = sympyopt2.bit_var(name="y_@@_OH_0")
        y2 = sympyopt2.bit_var(name="y_@@_OH_1")
        y3 = sympyopt2.bit_var(name="y_@@_OH_2")
        y4 = sympyopt2.bit_var(name="y_@@_OH_3")
        y5 = sympyopt2.bit_var(name="y_@@_OH_4")
        y6 = sympyopt2.bit_var(name="y_@@_OH_5")
        c = ConstraintEq(y1 + y2 + y3 + y4 + y5 + y6, 1)
        sympyopt2.add_constraint(c, name="_@@_OH_y")
        sympyopt2.minimize(2 * x + 6 * y1 + 3 * y2 - 3 * y4 - 6 * y5 - 9 * y6 + 2)
        assert sympyopt == sympyopt2

    def test_constraints(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        c = ConstraintEq(-1.5 * x + 2 * y, 4)
        sympyopt.add_constraint(c, "lin")
        conv = VarOneHot(sympyopt.variables["x"])
        sympyopt = conv.convert(sympyopt)

        sympyopt2 = SympyOpt()
        y = sympyopt2.int_var(lb=-2, ub=3, name="y")
        x1 = sympyopt2.bit_var(name="x_@@_OH_0")
        x2 = sympyopt2.bit_var(name="x_@@_OH_1")
        x3 = sympyopt2.bit_var(name="x_@@_OH_2")
        c = ConstraintEq(x1 + x2 + x3, 1)
        sympyopt2.add_constraint(c, name="_@@_OH_x")
        c = ConstraintEq(-1.5 * x2 - 3 * x3 + 2 * y, 4)
        sympyopt2.add_constraint(c, name="lin")
        assert sympyopt == sympyopt2
