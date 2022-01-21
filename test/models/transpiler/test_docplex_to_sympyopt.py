import pytest
from docplex.mp.model import Model

from omniqubo.models.sympyopt.constraints import INEQ_GEQ_SENSE, ConstraintEq, ConstraintIneq
from omniqubo.models.sympyopt.sympyopt import SympyOpt
from omniqubo.models.sympyopt.transpiler.docplex_to_sympyopt import DocplexToSympyopt
from omniqubo.models.sympyopt.vars import BitVar, IntVar, RealVar


class TestDocplexToSympyoptObjective:
    def test_zero_objective(self):
        mdl = Model(name="tsp")
        sympymodel = DocplexToSympyopt().transpile(mdl)
        assert sympymodel == SympyOpt()

    def test_const_objective(self):
        mdl = Model(name="tsp")
        mdl.minimize(2)
        sympymodel = DocplexToSympyopt().transpile(mdl)
        sympyopt = SympyOpt()
        sympyopt.minimize(2)
        assert sympymodel == sympyopt

    def test_max_const_objectives(self):
        mdl = Model(name="tsp")
        mdl.maximize(-1)
        sympymodel = DocplexToSympyopt().transpile(mdl)
        sympyopt = SympyOpt()
        sympyopt.maximize(-1)
        assert sympymodel == sympyopt

    def test_monomial_objective(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        mdl.minimize(10.5 * x)
        sympymodel = DocplexToSympyopt().transpile(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        sympyopt.minimize(10.5 * xx)
        assert sympymodel == sympyopt

    def test_linear_objective(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize(2 * x - 3 * y + 2)
        sympymodel = DocplexToSympyopt().transpile(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.minimize(2 * xx - 3 * yy + 2)
        assert sympymodel == sympyopt

    def test_quadratic_objective(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize((2 * x - 3 * y + 2) ** 2)
        sympymodel = DocplexToSympyopt().transpile(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.minimize((2 * xx - 3 * yy + 2) ** 2)
        assert sympymodel == sympyopt


class TestDocplexToSympyoptCanCheck:
    def test_int(self):
        mdl = Model(name="tsp")
        x = mdl.integer_var(-2, 4, "x")
        y = mdl.binary_var("y")
        z = mdl.continuous_var(-4, 3, "z")
        mdl.minimize((x + y + z) ** 2)
        mdl.add_constraint((2 * x + 4 * y + z) ** 2 <= 2)
        mdl.add_constraint(2 * x + 4 * y - z >= 0)
        mdl.add_constraint(2 * x + 4 * y == 0)
        DocplexToSympyopt().transpile(mdl)  # should not from an error
        assert DocplexToSympyopt().can_transpile(mdl)

    def test_semireal(self):
        mdl = Model(name="tsp")
        y = mdl.semicontinuous_var(lb=2, name="y")
        mdl.minimize(2 * y)

        assert not DocplexToSympyopt().can_transpile(mdl)
        with pytest.raises(NotImplementedError):
            DocplexToSympyopt().transpile(mdl)

    def test_semiint(self):
        mdl = Model(name="tsp")
        y = mdl.semiinteger_var(lb=2, name="y")
        mdl.minimize(2 * y)

        assert not DocplexToSympyopt().can_transpile(mdl)
        with pytest.raises(NotImplementedError):
            DocplexToSympyopt().transpile(mdl)


class TestDocplexToSympyoptTypes:
    def test_bit(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        mdl.minimize(2 * x)
        sympyopt = DocplexToSympyopt().transpile(mdl)
        assert sympyopt.variables["x"] == BitVar("x")

    def test_int(self):
        mdl = Model(name="tsp")
        x = mdl.integer_var(-2, 4, "x")
        mdl.minimize(2 * x)
        sympyopt = DocplexToSympyopt().transpile(mdl)
        assert sympyopt.variables["x"] == IntVar("x", -2, 4)

        mdl = Model(name="tsp")
        y = mdl.integer_var(lb=-2, name="y")
        mdl.minimize(2 * y)
        sympyopt = DocplexToSympyopt().transpile(mdl)
        assert sympyopt.variables["y"] == IntVar(lb=-2, ub=1e20, name="y")

    def test_real(self):
        mdl = Model(name="tsp")
        y = mdl.continuous_var(lb=-2.5, ub=3.1, name="y")
        mdl.minimize(2 * y)
        sympyopt = DocplexToSympyopt().transpile(mdl)
        assert sympyopt.variables["y"] == RealVar(lb=-2.5, ub=3.1, name="y")

        mdl = Model(name="tsp")
        y = mdl.continuous_var(name="y")
        mdl.minimize(2 * y)
        sympyopt = DocplexToSympyopt().transpile(mdl)
        assert sympyopt.variables["y"] == RealVar(lb=0, ub=1e20, name="y")

    def test_semireal(self):
        mdl = Model(name="tsp")
        y = mdl.semicontinuous_var(lb=2, name="y")
        mdl.minimize(2 * y)
        with pytest.raises(NotImplementedError):
            DocplexToSympyopt().transpile(mdl)

    def test_semiint(self):
        mdl = Model(name="tsp")
        y = mdl.semiinteger_var(lb=2, name="y")
        mdl.minimize(2 * y)
        with pytest.raises(NotImplementedError):
            DocplexToSympyopt().transpile(mdl)


class TestDocplexToSympyoptConstraints:
    def test_lineareq_constraints(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize(2 * x - 3 * y)
        mdl.add_constraint(2 * x + 3 * y == 2, ctname="lin1")
        mdl.add_constraint(x + 10.5 * y == 1.1, ctname="lin2")
        mdl.add_constraint(y == 5, ctname="trivial")
        sympymodel = DocplexToSympyopt().transpile(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.minimize(2 * xx - 3 * yy)
        sympyopt.add_constraint(ConstraintEq(2 * xx + 3 * yy, 2), name="lin1")
        sympyopt.add_constraint(ConstraintEq(xx + 10.5 * yy, 1.1), name="lin2")
        sympyopt.add_constraint(ConstraintEq(yy, 5), name="trivial")

        assert sympymodel == sympyopt

    def test_lineareq_commute_constraints(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize(2 * x - 3 * y)
        mdl.add_constraint(2 * x + 3 * y == 2, ctname="lin1")
        sympymodel = DocplexToSympyopt().transpile(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.minimize(2 * xx - 3 * yy)
        sympyopt.add_constraint(ConstraintEq(2, 2 * xx + 3 * yy), name="lin1")

        assert sympymodel == sympyopt

    def test_linearlineq_constraints(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize(2 * x - 3 * y)
        mdl.add_constraint(2 * x + 3 * y >= 2, ctname="lin1")
        mdl.add_constraint(x + 10.5 * y <= 1.1, ctname="lin2")
        mdl.add_constraint(y == 5, ctname="trivial")
        sympymodel = DocplexToSympyopt().transpile(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.minimize(2 * xx - 3 * yy)
        sympyopt.add_constraint(ConstraintIneq(2 * xx + 3 * yy, 2, INEQ_GEQ_SENSE), name="lin1")
        sympyopt.add_constraint(ConstraintIneq(xx + 10.5 * yy, 1.1), name="lin2")
        sympyopt.add_constraint(ConstraintEq(yy, 5), name="trivial")

        assert sympymodel == sympyopt

    def test_badineq_constraint(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.add_constraint(2 * x + 3 * y >= 2, ctname="lin1")
        sympymodel = DocplexToSympyopt().transpile(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.add_constraint(ConstraintIneq(2 * xx + 3 * yy, 2), name="lin1")
        with pytest.raises(AssertionError):
            assert sympymodel == sympyopt

    def test_quadeq_constraints(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize(2 * x - 3 * y)
        mdl.add_constraint(2 * x ** 2 + 3 * y == 2, ctname="quad1")
        mdl.add_constraint((x + 10.5 * y) ** 2 == 1.1, ctname="quad2")
        sympymodel = DocplexToSympyopt().transpile(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.minimize(2 * xx - 3 * yy)
        sympyopt.add_constraint(ConstraintEq(2 * xx ** 2 + 3 * yy, 2), name="quad1")
        sympyopt.add_constraint(ConstraintEq((xx + 10.5 * yy) ** 2, 1.1), name="quad2")

        assert sympymodel == sympyopt

    def test_quadineq_constraints(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize(2 * x - 3 * y)
        mdl.add_constraint(2 * x ** 2 + 3 * y >= 2, ctname="quad1")
        mdl.add_constraint(1.1 <= (x + 10.5 * y) ** 2, ctname="quad2")
        sympymodel = DocplexToSympyopt().transpile(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.minimize(2 * xx - 3 * yy)
        sympyopt.add_constraint(
            ConstraintIneq(2 * xx ** 2 + 3 * yy, 2, INEQ_GEQ_SENSE), name="quad1"
        )
        sympyopt.add_constraint(ConstraintIneq(1.1, (xx + 10.5 * yy) ** 2), name="quad2")

        assert sympymodel == sympyopt
