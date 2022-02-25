from sympy.core.evalf import INF

from omniqubo.models.sympyopt.sympyopt import SympyOpt


class TestIntVar:
    def test_eq(self):
        sympyopt1 = SympyOpt()
        sympyopt2 = SympyOpt()
        sympyopt3 = SympyOpt()
        sympyopt4 = SympyOpt()

        y1 = sympyopt1.variables[sympyopt1.int_var(name="y").name]
        y2 = sympyopt2.variables[sympyopt2.int_var(name="y").name]
        y3 = sympyopt3.variables[sympyopt3.int_var(name="y", lb=0, ub=4).name]
        y4 = sympyopt4.variables[sympyopt4.bit_var(name="y").name]

        assert y1 == y2
        assert y1 != y3
        assert y1 != y4

    def test_str(self):
        sympyopt = SympyOpt()
        y = sympyopt.variables[sympyopt.int_var(name="y").name]
        assert str(y) == "Integer -inf <= y <= inf"
        x = sympyopt.variables[sympyopt.int_var(name="x", lb=4, ub=10).name]
        assert str(x) == "Integer 4 <= x <= 10"

    def test_bounds(self):
        sympyopt = SympyOpt()
        y1 = sympyopt.variables[sympyopt.int_var(name="y1").name]
        y2 = sympyopt.variables[sympyopt.int_var(name="y2", lb=-1, ub=4).name]
        assert y1.get_lb() == -INF
        assert y1.get_ub() == INF
        assert y2.get_lb() == -1
        assert y2.get_ub() == 4


class TestBitVar:
    def test_eq(self):
        sympyopt1 = SympyOpt()
        sympyopt2 = SympyOpt()
        sympyopt3 = SympyOpt()

        y1 = sympyopt1.variables[sympyopt1.bit_var(name="y").name]
        y2 = sympyopt2.variables[sympyopt2.bit_var(name="y").name]
        y3 = sympyopt3.variables[sympyopt3.int_var(name="y").name]

        assert y1 == y2
        assert y1 != y3

    def test_bounds(self):
        sympyopt = SympyOpt()
        x = sympyopt.variables[sympyopt.bit_var(name="x").name]
        assert x.get_lb() == 0
        assert x.get_ub() == 1

    def test_str(self):
        sympyopt = SympyOpt()
        y = sympyopt.variables[sympyopt.bit_var(name="y").name]
        assert str(y) == "Bit y"


class TestRealVar:
    def test_eq(self):
        sympyopt1 = SympyOpt()
        sympyopt2 = SympyOpt()
        sympyopt3 = SympyOpt()
        sympyopt4 = SympyOpt()

        y1 = sympyopt1.variables[sympyopt1.real_var(name="y").name]
        y2 = sympyopt2.variables[sympyopt2.real_var(name="y").name]
        y3 = sympyopt3.variables[sympyopt3.real_var(name="y", lb=0, ub=4).name]
        y4 = sympyopt4.variables[sympyopt4.int_var(name="y").name]

        assert y1 == y2
        assert y1 != y3
        assert y1 != y4

    def test_bounds(self):
        sympyopt = SympyOpt()
        y1 = sympyopt.variables[sympyopt.real_var(name="y1").name]
        y2 = sympyopt.variables[sympyopt.real_var(name="y2", lb=-1, ub=4).name]
        assert y1.get_lb() == -INF
        assert y1.get_ub() == INF
        assert y2.get_lb() == -1
        assert y2.get_ub() == 4

    def test_str(self):
        sympyopt = SympyOpt()
        y = sympyopt.variables[sympyopt.real_var(name="y").name]
        assert str(y) == "Real -inf <= y <= inf"
        x = sympyopt.variables[sympyopt.real_var(name="x", lb=4, ub=10).name]
        assert str(x) == "Real 4 <= x <= 10"


class TestSpinVar:
    def test_eq(self):
        sympyopt1 = SympyOpt()
        sympyopt2 = SympyOpt()
        sympyopt3 = SympyOpt()

        y1 = sympyopt1.variables[sympyopt1.spin_var(name="y").name]
        y2 = sympyopt2.variables[sympyopt2.spin_var(name="y").name]
        y3 = sympyopt3.variables[sympyopt3.int_var(name="y").name]

        assert y1 == y2
        assert y1 != y3

    def test_str(self):
        sympyopt = SympyOpt()
        y = sympyopt.variables[sympyopt.spin_var(name="y").name]
        assert str(y) == "Spin y"

    def test_bounds(self):
        sympyopt = SympyOpt()
        x = sympyopt.variables[sympyopt.spin_var(name="x").name]
        assert x.get_lb() == -1
        assert x.get_ub() == 1
