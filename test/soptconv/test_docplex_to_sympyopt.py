import pytest
from docplex.mp.model import Model

from omniqubo.soptconv.docplex_to_sympyopt import DocplexToSymopt
from omniqubo.sympyopt import INEQ_GEQ_SENSE, INEQ_LEQ_SENSE, ConstraintEq, ConstraintIneq, SympyOpt


class TestDocplexToSymoptObjective:
    def test_zero_objective(self):
        mdl = Model(name="tsp")
        sympymodel = DocplexToSymopt().convert(mdl)
        assert sympymodel == SympyOpt()

    def test_const_objective(self):
        mdl = Model(name="tsp")
        mdl.minimize(2)
        sympymodel = DocplexToSymopt().convert(mdl)
        sympyopt = SympyOpt()
        sympyopt.minimize(2)
        assert sympymodel == sympyopt

    def test_max_const_objectives(self):
        mdl = Model(name="tsp")
        mdl.maximize(-1)
        sympymodel = DocplexToSymopt().convert(mdl)
        sympyopt = SympyOpt()
        sympyopt.maximize(-1)
        assert sympymodel == sympyopt

    def test_monomial_objective(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        mdl.minimize(10.5 * x)
        sympymodel = DocplexToSymopt().convert(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        sympyopt.minimize(10.5 * xx)
        assert sympymodel == sympyopt

    def test_linear_objective(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize(2 * x - 3 * y + 2)
        sympymodel = DocplexToSymopt().convert(mdl)

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
        sympymodel = DocplexToSymopt().convert(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.minimize((2 * xx - 3 * yy + 2) ** 2)
        assert sympymodel == sympyopt


class TestDocplexToSymoptConstraints:
    def test_lineareq_constraints(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize(2 * x - 3 * y)
        mdl.add_constraint(2 * x + 3 * y == 2, ctname="lin1")
        mdl.add_constraint(x + 10.5 * y == 1.1, ctname="lin2")
        mdl.add_constraint(y == 5, ctname="trivial")
        sympymodel = DocplexToSymopt().convert(mdl)

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
        sympymodel = DocplexToSymopt().convert(mdl)

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
        sympymodel = DocplexToSymopt().convert(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.minimize(2 * xx - 3 * yy)
        sympyopt.add_constraint(ConstraintIneq(2 * xx + 3 * yy, 2, INEQ_GEQ_SENSE), name="lin1")
        sympyopt.add_constraint(ConstraintIneq(xx + 10.5 * yy, 1.1, INEQ_LEQ_SENSE), name="lin2")
        sympyopt.add_constraint(ConstraintEq(yy, 5), name="trivial")

        assert sympymodel == sympyopt

    def test_badineq_constraint(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.add_constraint(2 * x + 3 * y >= 2, ctname="lin1")
        sympymodel = DocplexToSymopt().convert(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.add_constraint(ConstraintIneq(2 * xx + 3 * yy, 2, INEQ_LEQ_SENSE), name="lin1")
        with pytest.raises(AssertionError):
            assert sympymodel == sympyopt

    def test_quadeq_constraints(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize(2 * x - 3 * y)
        mdl.add_constraint(2 * x ** 2 + 3 * y == 2, ctname="quad1")
        mdl.add_constraint((x + 10.5 * y) ** 2 == 1.1, ctname="quad2")
        sympymodel = DocplexToSymopt().convert(mdl)

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
        sympymodel = DocplexToSymopt().convert(mdl)

        sympyopt = SympyOpt()
        xx = sympyopt.bit_var("x")
        yy = sympyopt.int_var(lb=-2, ub=10, name="y")
        sympyopt.minimize(2 * xx - 3 * yy)
        sympyopt.add_constraint(
            ConstraintIneq(2 * xx ** 2 + 3 * yy, 2, INEQ_GEQ_SENSE), name="quad1"
        )
        sympyopt.add_constraint(
            ConstraintIneq(1.1, (xx + 10.5 * yy) ** 2, INEQ_LEQ_SENSE), name="quad2"
        )

        assert sympymodel == sympyopt
