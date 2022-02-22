from dimod import BinaryQuadraticModel

from omniqubo.models.sympyopt import SympyOpt
from omniqubo.models.sympyopt.constraints import ConstraintEq
from omniqubo.models.sympyopt.transpiler.sympyopt_to_bqm import SympyOptToDimod


class TestSympyOptToDimod:
    def test_zero(self):
        sympyopt = SympyOpt()
        transpiler = SympyOptToDimod()
        assert transpiler.can_transpile(sympyopt)
        bqm_sym = transpiler.transpile(sympyopt)
        bqm = BinaryQuadraticModel(offset=0, vartype="BINARY")
        assert bqm_sym == bqm

    def test_constant(self):
        sympyopt = SympyOpt()
        sympyopt.minimize(2)
        transpiler = SympyOptToDimod()
        assert transpiler.can_transpile(sympyopt)
        bqm_sym = transpiler.transpile(sympyopt)
        bqm = BinaryQuadraticModel(offset=2, vartype="BINARY")
        assert bqm_sym == bqm

    def test_monomial(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var("x")
        sympyopt.minimize(2 * x)
        transpiler = SympyOptToDimod()
        assert transpiler.can_transpile(sympyopt)
        bqm_sym = transpiler.transpile(sympyopt)
        bqm = BinaryQuadraticModel({"x": 2}, {}, vartype="BINARY")
        assert bqm == bqm_sym

    def test_qubo(self):
        sympyopt = SympyOpt()
        x = sympyopt.bit_var("x")
        y = sympyopt.bit_var("y")
        z = sympyopt.bit_var("z")
        sympyopt.minimize((x + y) ** 2 + 2 * z - 3)
        transpiler = SympyOptToDimod()
        assert transpiler.can_transpile(sympyopt)
        bqm_sym = transpiler.transpile(sympyopt)
        bqm = BinaryQuadraticModel({"x": 1, "y": 1, "z": 2}, {("x", "y"): 2}, vartype="BINARY")
        bqm.offset += -3
        assert bqm == bqm_sym

        # bad one
        bqm = BinaryQuadraticModel({"x": 2, "y": 1, "z": 2}, {("x", "y"): 2}, vartype="BINARY")
        bqm.offset += -3
        assert bqm != bqm_sym

    def test_can_transpile(self):
        # maximization
        sympyopt = SympyOpt()
        sympyopt.maximize(2)
        transpiler = SympyOptToDimod()
        assert not transpiler.can_transpile(sympyopt)

        # hobo
        sympyopt = SympyOpt()
        x = sympyopt.bit_var("x")
        y = sympyopt.bit_var("y")
        z = sympyopt.bit_var("z")
        sympyopt.maximize(x * y * z)
        transpiler = SympyOptToDimod()
        assert not transpiler.can_transpile(sympyopt)

        # int var
        sympyopt = SympyOpt()
        x = sympyopt.int_var("x")
        y = sympyopt.bit_var("y")
        sympyopt.maximize(x * y)
        transpiler = SympyOptToDimod()
        assert not transpiler.can_transpile(sympyopt)

        # constrained
        sympyopt = SympyOpt()
        x = sympyopt.bit_var("x")
        y = sympyopt.bit_var("y")
        sympyopt.maximize(x * y)
        sympyopt.add_constraint(ConstraintEq(x, y))
        transpiler = SympyOptToDimod()
        assert not transpiler.can_transpile(sympyopt)
