from sympy import S, sin, sympify

from omniqubo.sympyopt import SympyOpt


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
