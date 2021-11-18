import pytest
from docplex.mp.model import Model

from omniqubo.omniqubo.omniqubo import Omniqubo
from omniqubo.sympyopt import SympyOpt
from omniqubo.sympyopt.constraints import ConstraintEq


class TestOmniquboInit:
    def test_docplex(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize((2 * x - 3 * y + 2) ** 2)
        omniqubo = Omniqubo(mdl)
        assert isinstance(omniqubo.model, SympyOpt)

    def test_sympyopt(self):
        sympyopt = SympyOpt()
        y1 = sympyopt.int_var(lb=0, ub=2, name="y1")
        y2 = sympyopt.int_var(lb=-2, ub=10, name="y20")
        x = sympyopt.bit_var(name="x")
        sympyopt.minimize(2 * y1 - 3 * y2 + x)
        omniqubo = Omniqubo(sympyopt)
        assert omniqubo.model == sympyopt

    def test_error(self):
        with pytest.raises(ValueError):
            Omniqubo(1)


class TestOmniqubo:
    def test_name_int_to_bits(self):
        sympyopt = SympyOpt()
        y1 = sympyopt.int_var(lb=0, ub=2, name="y1")
        y2 = sympyopt.int_var(lb=-2, ub=10, name="y20")
        x = sympyopt.bit_var(name="x")
        sympyopt.minimize(2 * y1 - 3 * y2 + x)
        sympyopt.add_constraint(ConstraintEq(2 * y1 + 3 * y2, 2), name="lin1")
        sympyopt.add_constraint(ConstraintEq(y1 + 10.5 * y2, 1.1), name="lin2")

        omniqubo = Omniqubo(sympyopt)
        omniqubo.int_to_bits(name="y1", mode="one-hot")
        assert not omniqubo.is_bm()
        omniqubo.int_to_bits(name="y20", mode="one-hot")
        assert omniqubo.is_bm()

        omniqubo = Omniqubo(sympyopt)
        assert not omniqubo.is_bm()
        omniqubo.int_to_bits(regname="y[0-9]+", mode="one-hot")
        assert omniqubo.is_bm()

        omniqubo = Omniqubo(sympyopt)
        z = sympyopt.int_var(lb=3, ub=6, name="z")
        sympyopt.add_constraint(ConstraintEq(y1 + 10.5 * y2, z), name="with z")
        assert not omniqubo.is_bm()
        omniqubo.int_to_bits(mode="one-hot")
        assert omniqubo.is_bm()

    def test_name_eq_to_obj(self):
        sympyopt = SympyOpt()
        y1 = sympyopt.int_var(lb=0, ub=2, name="y1")
        y2 = sympyopt.int_var(lb=-2, ub=10, name="y2")
        x = sympyopt.bit_var(name="x")
        sympyopt.minimize(2 * y1 - 3 * y2 + x)
        sympyopt.add_constraint(ConstraintEq(2 * y1 + 3 * y2, 2), name="lin1")
        sympyopt.add_constraint(ConstraintEq(y1 ** 2 + 10.5 * y2, 1.1), name="lin23")

        omniqubo = Omniqubo(sympyopt)
        omniqubo.eq_to_obj(name="lin1", penalty=10)
        assert len(omniqubo.model.constraints) == 1
        omniqubo.eq_to_obj(name="lin23", penalty=2.0)
        assert len(omniqubo.model.constraints) == 0

        omniqubo = Omniqubo(sympyopt)
        assert len(omniqubo.model.constraints) == 2
        omniqubo.eq_to_obj(regname="lin", penalty=1.0)
        assert len(omniqubo.model.constraints) == 2

        omniqubo.eq_to_obj(regname="lin[0-9]+", penalty=1.0)
        assert len(omniqubo.model.constraints) == 0
