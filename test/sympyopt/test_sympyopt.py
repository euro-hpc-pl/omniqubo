from sympy import S, sin, sympify

from omniqubo.sympyopt import SympyOpt
from omniqubo.sympyopt.constraints import INEQ_GEQ_SENSE, ConstraintEq, ConstraintIneq


class TestSympySympyOpt:
    def test_bitspin_simp_rec(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var("x")
        y = sympyopt.int_var(lb=-2, ub=10, name="y")
        z = sympyopt.spin_var("z")
        expr1 = sympyopt._bitspin_simp((x - y) ** 3)
        expr2 = x - 3 * x * y + 3 * x * y ** 2 - y ** 3
        assert sympify(expr1 - expr2) == 0

        expr1 = sympyopt._bitspin_simp((x - z) ** 3)
        expr2 = 4 * x - 3 * x * z - z
        assert sympify(expr1 - expr2) == 0

        expr1 = sympyopt._bitspin_simp(sin(x ** 3))
        expr2 = sin(x ** 3)
        assert sympify(expr1 - expr2) == 0

        expr1 = sympyopt._bitspin_simp(x ** 4)
        expr2 = x
        assert sympify(expr1 - expr2) == 0

        expr1 = sympyopt._bitspin_simp(z ** 4)
        expr2 = S(1)
        assert sympify(expr1 - expr2) == 0

    def test_qubo_isstatements(self):
        sopt = SympyOpt()
        x = sopt.bit_var("x")
        y = sopt.bit_var("y")
        z = sopt.bit_var("z")
        sopt.minimize(x ** 2 + (z + y) ** 2)

        assert sopt.is_bm()
        assert sopt.is_qubo()
        assert sopt.is_hobo()
        assert sopt.is_pp()
        assert sopt.is_qip()
        assert not sopt.is_lip()
        assert sopt.is_qcqp()
        assert not sopt.is_ising()

    def test_hobo_isstatements(self):
        sopt = SympyOpt()
        x = sopt.bit_var("x")
        y = sopt.bit_var("y")
        z = sopt.bit_var("z")
        sopt.minimize(x ** 2 + (z + y + x) ** 4)

        assert sopt.is_bm()
        assert not sopt.is_qubo()
        assert sopt.is_hobo()
        assert sopt.is_pp()
        assert not sopt.is_qip()
        assert not sopt.is_lip()
        assert not sopt.is_qcqp()
        assert not sopt.is_ising()
        assert not sopt.is_ising(locality=3)

    def test_ising_isstatements(self):
        sopt = SympyOpt()
        x = sopt.spin_var("x")
        y = sopt.spin_var("y")
        z = sopt.spin_var("z")
        sopt.minimize(x ** 2 + (z + y) ** 2)

        assert sopt.is_bm()
        assert not sopt.is_qubo()
        assert not sopt.is_hobo()
        assert not sopt.is_pp()
        assert not sopt.is_qip()
        assert not sopt.is_lip()
        assert not sopt.is_qcqp()
        assert sopt.is_ising()
        assert sopt.is_ising(locality=3)
        assert not sopt.is_ising(locality=1)

        sopt = SympyOpt()
        x = sopt.spin_var("x")
        y = sopt.spin_var("y")
        z = sopt.spin_var("z")
        sopt.minimize(x ** 2 + (z + y + x) ** 3)

        assert sopt.is_bm()
        assert not sopt.is_qubo()
        assert not sopt.is_hobo()
        assert not sopt.is_pp()
        assert not sopt.is_qip()
        assert not sopt.is_lip()
        assert not sopt.is_qcqp()
        assert not sopt.is_ising()
        assert sopt.is_ising(locality=3)
        assert not sopt.is_ising(locality=1)

    def test_qcpc_isstatements(self):
        sopt = SympyOpt()
        x = sopt.bit_var("x")
        y = sopt.int_var(lb=-2, ub=4, name="y")
        z = sopt.bit_var("z")
        sopt.minimize(2 * x + 4 * y - 3 * z ** 2)
        c = ConstraintIneq(2 * x - 3 * y + 3.4 * z, 1.4 + 2 * x, INEQ_GEQ_SENSE)
        sopt.add_constraint(c)
        c = ConstraintEq(2 * x - 3 * y + 3.4 * z ** 2, 1.4 + 2 * x)
        sopt.add_constraint(c)

        assert not sopt.is_bm()
        assert not sopt.is_qubo()
        assert not sopt.is_hobo()
        assert sopt.is_pp()
        assert sopt.is_qip()
        assert sopt.is_lip()
        assert sopt.is_qcqp()
        assert not sopt.is_ising()

        sopt.minimize(2 * x + 4 * y ** 2 - 3 * z ** 2)

        assert not sopt.is_bm()
        assert not sopt.is_qubo()
        assert not sopt.is_hobo()
        assert sopt.is_pp()
        assert sopt.is_qip()
        assert not sopt.is_lip()
        assert sopt.is_qcqp()
        assert not sopt.is_ising()

        c = ConstraintEq(2 * x - 3 * y ** 2 + 3.4 * z ** 2, 1.4 + 2 * x)
        sopt.add_constraint(c)

        assert not sopt.is_bm()
        assert not sopt.is_qubo()
        assert not sopt.is_hobo()
        assert sopt.is_pp()
        assert not sopt.is_qip()
        assert not sopt.is_lip()
        assert sopt.is_qcqp()
        assert not sopt.is_ising()

        sopt.minimize(2 * x + 4 * y ** 3 - 3 * z ** 2)

        assert not sopt.is_bm()
        assert not sopt.is_qubo()
        assert not sopt.is_hobo()
        assert sopt.is_pp()
        assert not sopt.is_qip()
        assert not sopt.is_lip()
        assert not sopt.is_qcqp()
        assert not sopt.is_ising()
