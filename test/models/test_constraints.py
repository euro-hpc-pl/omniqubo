import pytest

from omniqubo.models.sympyopt.constraints import INEQ_GEQ_SENSE, ConstraintEq, ConstraintIneq
from omniqubo.models.sympyopt.sympyopt import SympyOpt


class TestConstraintEq:
    def test_is_statemnts(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var("x")
        y = sympyopt.int_var("y")
        z = sympyopt.real_var("z")

        c = ConstraintEq(x + 2 * y - 4 * z, 3.1 - 2.4 * y + 3.4 * z)
        assert c.is_eq_constraint()
        assert not c.is_ineq_constraint()

    def test_eq(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var("x")
        y = sympyopt.int_var("y")
        z = sympyopt.real_var("z")

        c = ConstraintEq(x + 2 * y - 4 * z, 3.1 - 2.4 * y + 3.4 * z)
        assert c is not (x + 2 * y - 4 * z != 3.1 - 2.4 * y + 3.4 * z)

    def test_str(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var("x")
        y = sympyopt.int_var("y")
        z = sympyopt.real_var("z")

        c = ConstraintEq(x + 2 * y - 4 * z, 3.1 - 2.4 * y + 3.4 * z)
        assert str(c) == "x + 2*y - 4*z == -2.4*y + 3.4*z + 3.1"


class TestConstraintIneq:
    def test_is_statemnts(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var("x")
        y = sympyopt.int_var("y")
        z = sympyopt.real_var("z")

        c = ConstraintIneq(x + 2 * y - 4 * z, 3.1 - 2.4 * y + 3.4 * z, INEQ_GEQ_SENSE)
        assert not c.is_eq_constraint()
        assert c.is_ineq_constraint()

        c = ConstraintIneq(x + 2 * y - 4 * z, 3.1 - 2.4 * y + 3.4 * z)
        assert not c.is_eq_constraint()
        assert c.is_ineq_constraint()

    def test_init(self):
        with pytest.raises(ValueError):
            sympyopt = SympyOpt()
            x = sympyopt.bit_var("x")

            ConstraintIneq(x, 2, "bad string")

    def test_eq(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var("x")
        y = sympyopt.int_var("y")
        z = sympyopt.real_var("z")

        c = ConstraintIneq(x + 2 * y - 4 * z, 3.1 - 2.4 * y + 3.4 * z)
        assert c is not (x + 2 * y - 4 * z <= 3.1 - 2.4 * y + 3.4 * z)

    def test_str(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var("x")
        y = sympyopt.int_var("y")
        z = sympyopt.real_var("z")

        c = ConstraintIneq(x + 2 * y - 4 * z, 3.1 - 2.4 * y + 3.4 * z)
        assert str(c) == "x + 2*y - 4*z <= -2.4*y + 3.4*z + 3.1"
