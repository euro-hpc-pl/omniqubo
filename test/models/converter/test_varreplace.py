from dimod import ExactSolver
from pandas import DataFrame

from omniqubo.converters.eq_to_objective import EqToObj
from omniqubo.converters.varreplace import (
    BitToSpin,
    TrivialIntToBit,
    VarBinary,
    VarOneHot,
    interpret,
)
from omniqubo.models.sympyopt.constraints import ConstraintEq
from omniqubo.models.sympyopt.converters import convert
from omniqubo.models.sympyopt.sympyopt import SympyOpt
from omniqubo.models.sympyopt.transpiler.sympyopt_to_dimod import SympyOptToDimod
from omniqubo.sampleset._dimod_import import dimod_import


class TestOneHot:
    def test_objective(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.minimize(2 * x - 3 * y + 2)
        conv = VarOneHot("y", True)
        sympyopt = convert(sympyopt, conv)

        sympyopt2 = SympyOpt()
        x = sympyopt2.int_var(name="x", lb=0, ub=2)
        y1 = sympyopt2.bit_var(name="y___OH_0")
        y2 = sympyopt2.bit_var(name="y___OH_1")
        y3 = sympyopt2.bit_var(name="y___OH_2")
        y4 = sympyopt2.bit_var(name="y___OH_3")
        y5 = sympyopt2.bit_var(name="y___OH_4")
        y6 = sympyopt2.bit_var(name="y___OH_5")
        c = ConstraintEq(y1 + y2 + y3 + y4 + y5 + y6, 1)
        sympyopt2.add_constraint(c, name="___OH_y")
        sympyopt2.minimize(2 * x + 6 * y1 + 3 * y2 - 3 * y4 - 6 * y5 - 9 * y6 + 2)
        assert sympyopt == sympyopt2

    def test_constraints(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        c = ConstraintEq(-1.5 * x + 2 * y, 4)
        sympyopt.add_constraint(c, "lin")
        conv = VarOneHot("x", is_regexp=False)
        sympyopt = convert(sympyopt, conv)

        sympyopt2 = SympyOpt()
        y = sympyopt2.int_var(lb=-2, ub=3, name="y")
        x1 = sympyopt2.bit_var(name="x___OH_0")
        x2 = sympyopt2.bit_var(name="x___OH_1")
        x3 = sympyopt2.bit_var(name="x___OH_2")
        c = ConstraintEq(x1 + x2 + x3, 1)
        sympyopt2.add_constraint(c, name="___OH_x")
        c = ConstraintEq(-1.5 * x2 - 3 * x3 + 2 * y, 4)
        sympyopt2.add_constraint(c, name="lin")
        assert sympyopt == sympyopt2

    def test_interpret(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var(name="x")
        y1 = sympyopt.int_var(lb=-2, ub=3, name="y1")
        y2 = sympyopt.int_var(lb=0, ub=1, name="y2")
        sympyopt.minimize(2 * x - 3 * y1 + y2 + 2)

        conv1 = VarOneHot(".*", True)
        conv2 = EqToObj(".*", True, 2)
        sympyopt = convert(sympyopt, conv1)
        sympyopt = convert(sympyopt, conv2)

        bqm = SympyOptToDimod().transpile(sympyopt)
        samples = ExactSolver().sample_qubo(bqm.to_qubo()[0])
        samples = dimod_import(samples)
        print(samples)
        samples: DataFrame = interpret(samples, conv2)
        samples: DataFrame = interpret(samples, conv1)
        print(samples)
        assert sum(samples["feasible"]) == 24
        assert samples.shape[0] == 512  # because y2 is also encoded as one-hot
        samples = samples.loc[samples["feasible"]]
        assert set(samples["x"]) == {0, 1}
        assert set(samples["y1"]) == {-2, -1, 0, 1, 2, 3}
        assert set(samples["y2"]) == {0, 1}


class TestBinary:
    def test_objective(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=-1, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        sympyopt.minimize(2 * x - 3 * y + 2)
        conv = VarBinary(".*", True)
        sympyopt = convert(sympyopt, conv)

        sympyopt2 = SympyOpt()
        x1 = sympyopt2.bit_var(name="x___BIN_0")
        x2 = sympyopt2.bit_var(name="x___BIN_1")
        y1 = sympyopt2.bit_var(name="y___BIN_0")
        y2 = sympyopt2.bit_var(name="y___BIN_1")
        y3 = sympyopt2.bit_var(name="y___BIN_2")
        sympyopt2.minimize(2 * (-1 + x1 + 2 * x2) - 3 * (-2 + y1 + 2 * y2 + 2 * y3) + 2)
        print(sympyopt)
        print(sympyopt2)
        assert sympyopt == sympyopt2

    def test_constraints(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=-1, ub=2)
        y = sympyopt.int_var(lb=-2, ub=3, name="y")
        c = ConstraintEq(-1.5 * x + 2 * y, 4)
        sympyopt.add_constraint(c, "lin")
        conv = VarBinary("x", is_regexp=False)
        sympyopt = convert(sympyopt, conv)

        sympyopt2 = SympyOpt()
        y = sympyopt2.int_var(lb=-2, ub=3, name="y")
        x1 = sympyopt2.bit_var(name="x___BIN_0")
        x2 = sympyopt2.bit_var(name="x___BIN_1")
        c = ConstraintEq(-1.5 * (-1 + x1 + 2 * x2) + 2 * y, 4)
        sympyopt2.add_constraint(c, name="lin")
        assert sympyopt == sympyopt2

    def test_interpret(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var(name="x")
        y1 = sympyopt.int_var(lb=-2, ub=3, name="y1")
        y2 = sympyopt.int_var(lb=0, ub=1, name="y2")
        sympyopt.minimize(2 * x - 3 * y1 + y2 + 2)

        conv = VarBinary(".*", True)
        sympyopt = convert(sympyopt, conv)

        bqm = SympyOptToDimod().transpile(sympyopt)
        samples = ExactSolver().sample_qubo(bqm.to_qubo()[0])
        samples = dimod_import(samples)
        print(samples)
        samples: DataFrame = interpret(samples, conv)
        print(samples)
        assert sum(samples["feasible"]) == 2 ** 5  # some cases may be repetead
        assert samples.shape[0] == 2 ** 5
        samples = samples.loc[samples["feasible"]]
        assert set(samples["x"]) == {0, 1}
        assert set(samples["y1"]) == {-2, -1, 0, 1, 2, 3}
        assert set(samples["y2"]) == {0, 1}


class TestTrivialIntToBit:
    def test_conversion(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=1)
        y = sympyopt.int_var(name="y", lb=2, ub=3)
        c = ConstraintEq(y, 2)
        sympyopt.minimize(2 * x - 3 * y + 2)
        sympyopt.add_constraint(c, "c1")
        conv = TrivialIntToBit(".*", True)
        sympyopt = convert(sympyopt, conv)

        sympyopt2 = SympyOpt()
        xx = sympyopt2.bit_var(name="x___itb")
        yy = sympyopt2.bit_var(name="y___itb")
        cc = ConstraintEq(yy, 0)
        sympyopt2.add_constraint(cc, "c1")
        sympyopt2.minimize(2 * xx - 3 * (yy + 2) + 2)
        assert sympyopt == sympyopt2

    def test_interpret(self):
        sympyopt = SympyOpt()
        x = sympyopt.int_var(name="x", lb=0, ub=1)
        y = sympyopt.int_var(name="y", lb=2, ub=3)
        sympyopt.minimize(2 * x - 3 * y + 2)
        conv = TrivialIntToBit(".*", True)
        sympyopt = convert(sympyopt, conv)

        bqm = SympyOptToDimod().transpile(sympyopt)
        samples = ExactSolver().sample_qubo(bqm.to_qubo()[0])
        samples = dimod_import(samples)
        samples: DataFrame = interpret(samples, conv)
        print(samples)

        assert samples.shape[0] == 4
        assert set(samples["y"]) == {2, 3}
        assert set(samples["x"]) == {0, 1}


class TestBitToSpin:
    def test_conversion(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var(name="x")
        y = sympyopt.bit_var(name="y")
        c = ConstraintEq(y, 2)
        sympyopt.minimize(2 * x * y - x + 2 * y + 2)
        sympyopt.add_constraint(c, "c1")
        conv = BitToSpin(".*", True, True)
        sympyopt = convert(sympyopt, conv)

        sympyopt2 = SympyOpt()
        xx = sympyopt2.spin_var(name="x___bts")
        yy = sympyopt2.spin_var(name="y___bts")
        cc = ConstraintEq((1 - yy) / 2, 2)
        sympyopt2.add_constraint(cc, "c1")
        sympyopt2.minimize((1 - xx) * (1 - yy) / 2 - (1 - xx) / 2 + 1 - yy + 2)
        assert sympyopt == sympyopt2

    def test_conversion_not_reversed(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var(name="x")
        y = sympyopt.bit_var(name="y")
        c = ConstraintEq(y, 2)
        sympyopt.minimize(2 * x * y - x + 2 * y + 2)
        sympyopt.add_constraint(c, "c1")
        conv = BitToSpin(".*", True, False)
        sympyopt = convert(sympyopt, conv)

        sympyopt2 = SympyOpt()
        xx = sympyopt2.spin_var(name="x___bts")
        yy = sympyopt2.spin_var(name="y___bts")
        cc = ConstraintEq((1 + yy) / 2, 2)
        sympyopt2.add_constraint(cc, "c1")
        sympyopt2.minimize((1 + xx) * (1 + yy) / 2 - (1 + xx) / 2 + 1 + yy + 2)
        assert sympyopt == sympyopt

    def test_no_regexp(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var(name="x")
        y = sympyopt.bit_var(name="y")
        c = ConstraintEq(y, 2)
        sympyopt.add_constraint(c, "c1")
        sympyopt.minimize(2 * x * y - x + 2 * y + 2)
        conv = BitToSpin("y", False, True)
        convert(sympyopt, conv)

        sympyopt2 = SympyOpt()
        x = sympyopt2.bit_var(name="x")
        yy = sympyopt2.spin_var(name="y___bts")
        cc = ConstraintEq((1 - yy) / 2, 2)
        sympyopt2.add_constraint(cc, "c1")
        sympyopt2.minimize(2 * x * (1 - yy) / 2 - x + 1 - yy + 2)
        print(sympyopt)
        print(sympyopt2)
        assert sympyopt == sympyopt2

    def test_interpret(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var(name="x")
        y = sympyopt.bit_var(name="y")
        sympyopt.minimize(2 * x - 3 * y + 2)
        conv = BitToSpin(".*", True, True)
        sympyopt = convert(sympyopt, conv)

        bqm = SympyOptToDimod().transpile(sympyopt)
        h, J, _ = bqm.to_ising()
        samples = ExactSolver().sample_ising(h, J)
        samples = dimod_import(samples)
        print(samples)
        samples: DataFrame = interpret(samples, conv)
        print(samples)

        assert samples.shape[0] == 4
        assert set(samples["y"]) == {0, 1}
        assert set(samples["x"]) == {0, 1}
