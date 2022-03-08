import sys

from dimod import ExactSolver
from docplex.mp.model import Model

sys.path.append("/home/adam/Documents/omniqubo/")

from omniqubo import Omniqubo  # noqa: E402
from omniqubo.sampleset import dimod_import  # noqa: E402

# construct the model in language you like
mdl = Model("ILP")
x = mdl.integer_var(name="x", lb=-2, ub=2)
y = mdl.binary_var("y")
z = mdl.integer_var(name="z", lb=0, ub=4)
mdl.minimize((x + y) ** 2 - 2 * z + 3)
mdl.add_constraint(x >= 2 * y, ctname="c1")

# transform into QUBO
omniqubo = Omniqubo(mdl)
omniqubo.ineq_to_eq(".*")  # note to_qubo should be used here, but quadratize is required
omniqubo.int_to_bits(".*", mode="one-hot", is_regexp=True)
omniqubo.eq_to_obj(".*", penalty=20)
print(omniqubo.model)
print("Is QUBO: ", omniqubo.is_qubo())

# ...

# export the QUBO and solve it with third-party solver
bqm = omniqubo.export("bqm")
Q, offset = bqm.to_qubo()
df = ExactSolver().sample_qubo(Q)

# interpret the results
samples = omniqubo.interpret(dimod_import(df))
samples["energy"] += offset  # this needs to be done as solver ignores offset

# some samples analysis and tests
print("All samples:")
print(samples)
print("Feasible samples:")
print(samples.loc[samples["feasible"]])
