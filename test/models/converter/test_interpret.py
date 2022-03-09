from dimod import ExactSolver
from docplex.mp.model import Model

from omniqubo import Omniqubo
from omniqubo.sampleset import dimod_import


class TestUseCases:
    def test_docplex_qip_qubo(self):
        # construct the model in language you like
        mdl = Model("ILP")
        x = mdl.integer_var(name="x", lb=-2, ub=2)
        y = mdl.binary_var("y")
        z = mdl.integer_var(name="z", lb=0, ub=4)
        mdl.minimize((x + y) ** 2 - 2 * z + 3)
        mdl.add_constraint(x == 2 * y, ctname="c1")

        # transform into QUBO
        omniqubo = Omniqubo(mdl)
        omniqubo.int_to_bits(".*", "binary")
        omniqubo.eq_to_obj(".*", penalty=20)
        print(omniqubo.model)

        assert omniqubo.is_qubo()

        # export the QUBO and solve it with third-party solver
        bqm = omniqubo.export("dimod_bqm")
        Q, offset = bqm.to_qubo()
        df = ExactSolver().sample_qubo(Q)

        # interpret the results
        samples = omniqubo.interpret(dimod_import(df))
        samples["energy"] += offset  # this needs to be done as solver ignores offset

        # some samples analysis and tests
        assert samples.shape[0] == 2 ** 7
        samples = samples.loc[samples["feasible"]]

        # existance of this column "energy" may depend on the solver outcome
        samples = samples.sort_values("energy")
        best_sample = samples.iloc[0, :]
        assert best_sample["energy"] == -5
        assert best_sample["z"] == 4
        assert best_sample["x"] == 0
        assert best_sample["y"] == 0

    def test_docplex_qip_ising(self):
        mdl = Model("ILP")
        x = mdl.integer_var(name="x", lb=-2, ub=2)
        y = mdl.binary_var("y")
        z = mdl.integer_var(name="z", lb=0, ub=4)
        mdl.minimize((x + y) ** 2 - 2 * z + 3)
        mdl.add_constraint(x == 2 * y, ctname="c1")

        omniqubo = Omniqubo(mdl)
        omniqubo.int_to_bits(".*", "one-hot")
        omniqubo.eq_to_obj(".*", penalty=20)
        omniqubo.bit_to_spin(".*", reversed=True)

        assert omniqubo.is_ising(locality=2)

        bqm = omniqubo.export("dimod_bqm")
        h, J, offset = bqm.to_ising()
        df = ExactSolver().sample_ising(h, J)
        samples = omniqubo.interpret(dimod_import(df))
        samples["energy"] += offset
        assert samples.shape[0] == 2 * (2 ** 5) * (2 ** 5)

        samples = samples.loc[samples["feasible"]]
        assert samples.shape[0] == 10

        samples = samples.sort_values("energy")
        best_sample = samples.iloc[0, :]

        # energy may be different, but minimum should match
        # assert best_sample["energy"] == -5
        assert best_sample["z"] == 4
        assert best_sample["x"] == 0
        assert best_sample["y"] == 0

    def test_docplex_qip_qubo_ineq(self):
        # construct the model in language you like
        mdl = Model("ILP")
        x = mdl.integer_var(name="x", lb=-2, ub=2)
        y = mdl.binary_var("y")
        z = mdl.integer_var(name="z", lb=0, ub=4)
        mdl.minimize((x + y) ** 2 - 2 * z + 3)
        mdl.add_constraint(y <= x, ctname="c1")

        # transform into QUBO (without interpreting slacks)
        omniqubo = Omniqubo(mdl)
        omniqubo.ineq_to_eq(".*")
        omniqubo.eq_to_obj(".*")
        omniqubo.int_to_bits(".*", "binary")
        print(omniqubo.model)

        assert omniqubo.is_qubo()

        # export the QUBO and solve it with third-party solver
        bqm = omniqubo.export("dimod_bqm")
        Q, offset = bqm.to_qubo()
        df = ExactSolver().sample_qubo(Q)

        # interpret the results
        samples = omniqubo.interpret(dimod_import(df))
        samples["energy"] += offset  # this needs to be done as solver ignores offset

        # some samples analysis and tests
        assert samples.shape[0] == 2 ** 9

        samples = samples.loc[samples["feasible"]]
        assert samples.shape[0] > 25  # because slack value does not affect feasibility

        # existance of this column "energy" may depend on the solver outcome
        samples = samples.sort_values("energy")
        best_sample = samples.iloc[0, :]
        assert best_sample["energy"] == -5
        assert best_sample["z"] == 4
        assert best_sample["x"] == 0
        assert best_sample["y"] == 0

    def test_docplex_qip_qubo_ineq_slack_interpret(self):
        # construct the model in language you like
        mdl = Model("ILP")
        x = mdl.integer_var(name="x", lb=-2, ub=2)
        y = mdl.binary_var("y")
        z = mdl.integer_var(name="z", lb=0, ub=4)
        mdl.minimize((x + y) ** 2 - 2 * z + 3)
        mdl.add_constraint(y <= x, ctname="c1")
        mdl.add_constraint(z >= x, ctname="c2")

        # transform into QUBO (with interpreting slacks)
        omniqubo = Omniqubo(mdl)
        omniqubo.ineq_to_eq(".*", check_slack=True)
        omniqubo.eq_to_obj(".*")
        omniqubo.int_to_bits(".*", "binary")
        print(omniqubo.model)

        assert omniqubo.is_qubo()

        # export the QUBO and solve it with third-party solver
        bqm = omniqubo.export("dimod_bqm")
        Q, offset = bqm.to_qubo()
        df = ExactSolver().sample_qubo(Q)

        # interpret the results
        samples = omniqubo.interpret(dimod_import(df))
        samples["energy"] += offset  # this needs to be done as solver ignores offset

        # some samples analysis and tests
        assert samples.shape[0] == 2 ** 12
        samples = samples.loc[samples["feasible"]]

        # existance of this column "energy" may depend on the solver outcome
        samples = samples.sort_values("energy")
        best_sample = samples.iloc[0, :]
        assert best_sample["energy"] == -5
        assert best_sample["z"] == 4
        assert best_sample["x"] == 0
        assert best_sample["y"] == 0

        samples.pop("feasible")
        samples.pop("energy")
        assert samples.drop_duplicates().shape[0] == 19  # when repetitions dropped
