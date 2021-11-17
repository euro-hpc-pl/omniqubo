from omniqubo.omniqubo.omniqubo import Omniqubo
from omniqubo.sympyopt import SympyOpt
from omniqubo.sympyopt.constraints import ConstraintEq


class TestOmniqubo:
    def test_name_var(self):
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
