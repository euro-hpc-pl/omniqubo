from copy import deepcopy

import pytest
from pandas import DataFrame

from omniqubo import Omniqubo
from omniqubo.converters.converter import ConverterAbs, can_convert, convert, interpret
from omniqubo.models.sympyopt.constraints import ConstraintEq, ConstraintIneq
from omniqubo.models.sympyopt.sympyopt import SympyOpt


class TestNewConvert:
    def test_multiply_by(self):
        class MultiplyBy(ConverterAbs):
            def __init__(self, val: float) -> None:
                super().__init__()
                self.val = val

        @convert.register
        def convert_mul_by(model: SympyOpt, converter: MultiplyBy) -> SympyOpt:
            assert can_convert(model, converter)
            model.objective = converter.val * model.objective
            return model

        @can_convert.register
        def can_convert_mul_by(model: SympyOpt, converter: MultiplyBy) -> bool:
            return converter.val > 0

        sympyopt = SympyOpt()
        y1 = sympyopt.int_var(lb=0, ub=2, name="y1")
        y2 = sympyopt.int_var(lb=-2, ub=10, name="y20")
        x = sympyopt.bit_var(name="x")
        sympyopt.minimize(2 * y1 - 3 * y2 + x)
        sympyopt.add_constraint(ConstraintEq(2 * y1 + 3 * y2, 2), name="lin1")
        sympyopt.add_constraint(ConstraintIneq(y1 + 10.5 * y2, 1.1), name="lin2")

        sympyopt_ref = deepcopy(sympyopt)
        sympyopt_ref.minimize(4 * y1 - 6 * y2 + 2 * x)

        conv = MultiplyBy(2.0)

        omni = Omniqubo(sympyopt)
        omni.convert(conv)
        assert omni.model == sympyopt_ref

        conv = MultiplyBy(-1)
        with pytest.raises(AssertionError):
            omni.convert(conv)

    def test_missing_functions(self):
        class MultiplyObjBy(ConverterAbs):
            def __init__(self, val: float) -> None:
                super().__init__()
                self.val = val

        sympyopt = SympyOpt()
        x = sympyopt.bit_var(name="x")
        sympyopt.minimize(x)
        conv = MultiplyObjBy(-2)
        samples = DataFrame()

        with pytest.raises(NotImplementedError):
            convert(sympyopt, conv)

        with pytest.raises(NotImplementedError):
            interpret(samples, conv)

        with pytest.raises(NotImplementedError):
            assert can_convert(sympyopt, conv)
