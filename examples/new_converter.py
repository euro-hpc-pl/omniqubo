import sys

from dimod import ExactSolver
from docplex.mp.model import Model
from pandas import DataFrame

sys.path.append("/home/adam/Documents/omniqubo/")


from omniqubo import Omniqubo  # noqa: E402
from omniqubo.converters.converter import (  # noqa: E402
    ConverterAbs,
    can_convert,
    convert,
    interpret,
)
from omniqubo.models.sympyopt import SympyOpt  # noqa: E402
from omniqubo.sampleset import dimod_import  # noqa: E402


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


@interpret.register
def interpret_mul_by(samples: DataFrame, converter: MultiplyBy) -> bool:
    if "energy" in samples.columns:
        samples["energy"] /= converter.val
    return samples


# construct the model in language you like
mdl = Model("ILP")
x = mdl.integer_var(name="x", lb=-2, ub=2)
y = mdl.binary_var("y")
mdl.minimize((x + y) ** 2 + 3)

omniqubo = Omniqubo(mdl)
omniqubo.to_hobo(penalty=1)
omniqubo.convert(MultiplyBy(2.0))

# export the QUBO and solve it with third-party solver
bqm = omniqubo.export("bqm")
Q, offset = bqm.to_qubo()
df = ExactSolver().sample_qubo(Q)

# interpret the results
df = dimod_import(df)
df["energy"] += offset  # this needs to be done as solver ignores offset
print(df["energy"])
df = omniqubo.interpret(df)
print(df["energy"])
