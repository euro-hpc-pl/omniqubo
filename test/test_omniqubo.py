from copy import deepcopy

import pytest
from dimod import ExactSolver
from docplex.mp.model import Model
from sympy import sin

from omniqubo import Omniqubo
from omniqubo.models.sympyopt.constraints import INEQ_GEQ_SENSE, ConstraintEq, ConstraintIneq
from omniqubo.models.sympyopt.sympyopt import SympyOpt
from omniqubo.sampleset import dimod_import


class TestOmniquboInit:
    def test_docplex(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize((2 * x - 3 * y + 2) ** 2)
        omniqubo = Omniqubo(mdl)
        assert isinstance(omniqubo.model, SympyOpt)

    def test_sympyopt(self):
        sympyopt = SympyOpt()
        y1 = sympyopt.int_var(lb=0, ub=2, name="y1")
        y2 = sympyopt.int_var(lb=-2, ub=10, name="y20")
        x = sympyopt.bit_var(name="x")
        sympyopt.minimize(2 * y1 - 3 * y2 + x)
        omniqubo = Omniqubo(sympyopt)
        assert omniqubo.model == sympyopt

    def test_error(self):
        with pytest.raises(ValueError):
            Omniqubo(1)

    def test_logs(self):
        mdl = Model(name="tsp")
        x = mdl.binary_var("x")
        y = mdl.integer_var(lb=-2, ub=10, name="y")
        mdl.minimize((2 * x - 3 * y + 2) ** 2)

        omniqubo = Omniqubo(mdl, verbatim_logs=False)
        assert len(omniqubo.model_logs) == 0

        omniqubo = Omniqubo(mdl, verbatim_logs=True)
        assert len(omniqubo.model_logs) == 1
        omniqubo.int_to_bits(".*", "one-hot", trivial_conv=False)
        assert len(omniqubo.model_logs) == 2


class TestOmniqubo:
    def test_name_int_to_bits(self):
        sympyopt = SympyOpt()
        y1 = sympyopt.int_var(lb=0, ub=2, name="y1")
        y2 = sympyopt.int_var(lb=-2, ub=10, name="y20")
        x = sympyopt.bit_var(name="x")
        sympyopt.minimize(2 * y1 - 3 * y2 + x)
        sympyopt.add_constraint(ConstraintEq(2 * y1 + 3 * y2, 2), name="lin1")
        sympyopt.add_constraint(ConstraintEq(y1 + 10.5 * y2, 1.1), name="lin2")

        omniqubo = Omniqubo(sympyopt)
        omniqubo.int_to_bits(names="y1", mode="one-hot", is_regexp=False)
        assert not omniqubo.is_bm()
        omniqubo.int_to_bits(names="y20", mode="one-hot", is_regexp=False)
        assert omniqubo.is_bm()

        omniqubo = Omniqubo(sympyopt)
        assert not omniqubo.is_bm()
        omniqubo.int_to_bits(names="y[0-9]+", mode="one-hot")
        assert omniqubo.is_bm()

        omniqubo = Omniqubo(sympyopt)
        z = sympyopt.int_var(lb=3, ub=6, name="z")
        sympyopt.add_constraint(ConstraintEq(y1 + 10.5 * y2, z), name="with z")
        assert not omniqubo.is_bm()
        omniqubo.int_to_bits(".*", mode="one-hot")
        assert omniqubo.is_bm()

    def test_name_minmax(self):
        sympyopt = SympyOpt()
        y = sympyopt.int_var(lb=0, ub=2, name="y1")
        x = sympyopt.bit_var(name="x")
        sympyopt.minimize(2 * y + x)
        sympyopt.add_constraint(ConstraintEq(2 * y, 2), name="lin1")
        sympyopt.add_constraint(ConstraintEq(y, 1.1 + x), name="lin2")

        sympyopt_min = deepcopy(sympyopt)
        sympyopt.maximize(-2 * y - x)
        sympyopt_max = deepcopy(sympyopt)
        assert sympyopt_max != sympyopt_min

        omniqubo = Omniqubo(sympyopt_min)
        assert omniqubo.model == sympyopt_min
        omniqubo.make_max()
        assert omniqubo.model == sympyopt_max
        omniqubo.make_min()
        assert omniqubo.model == sympyopt_min

    def test_remove_constraint(self):
        sympyopt = SympyOpt()
        y = sympyopt.int_var(lb=0, ub=2, name="y1")
        x = sympyopt.bit_var(name="x")
        sympyopt.minimize(2 * y + x)
        sympyopt.add_constraint(ConstraintEq(2 * y, 2), name="lin1")
        sympyopt.add_constraint(ConstraintEq(y, 1.1 + x), name="lin2")

        omniqubo = Omniqubo(deepcopy(sympyopt))
        omniqubo.rm_constraints(names="lin1", is_regexp=False)
        assert omniqubo.model.constraints.keys() == {"lin2"}
        assert omniqubo.model.objective != 0

        omniqubo = Omniqubo(deepcopy(sympyopt))
        omniqubo.rm_constraints(names="lin[1-2]")
        assert len(omniqubo.model.constraints.keys()) == 0
        assert omniqubo.model.objective != 0

        omniqubo = Omniqubo(deepcopy(sympyopt))
        omniqubo.rm_constraints(names=".*")
        assert len(omniqubo.model.constraints.keys()) == 0
        assert omniqubo.model.objective != 0

    def test_docplex_rm_test_interpret(self):
        mdl = Model("ILP")
        x = mdl.binary_var(name="x")
        y = mdl.binary_var("y")
        z = mdl.binary_var(name="z")
        mdl.minimize((x + y) ** 2 - 2 * z + 3)
        mdl.add_constraint(x == 1, ctname="c1")
        mdl.add_constraint(y <= z, ctname="c2")
        mdl.add_constraint(x >= z, ctname="c3")

        omniqubo = Omniqubo(mdl)
        omniqubo.rm_constraints(".*")

        Q, offset = omniqubo.export("bqm").to_qubo()
        df = ExactSolver().sample_qubo(Q)
        samples = omniqubo.interpret(dimod_import(df))
        assert samples.shape[0] == 2 ** 3

        samples = samples.loc[samples["feasible"]]
        assert samples.shape[0] == 2 ** 3

        omniqubo = Omniqubo(mdl)
        omniqubo.rm_constraints(".*", check_constraints=True)

        Q, offset = omniqubo.export("bqm").to_qubo()
        df = ExactSolver().sample_qubo(Q)
        samples = omniqubo.interpret(dimod_import(df))
        assert samples.shape[0] == 2 ** 3
        samples = samples.loc[samples["feasible"]]
        assert samples.shape[0] == 3

    def test_name_eq_to_obj(self):
        sympyopt = SympyOpt()
        y1 = sympyopt.int_var(lb=0, ub=2, name="y1")
        y2 = sympyopt.int_var(lb=-2, ub=10, name="y2")
        x = sympyopt.bit_var(name="x")
        sympyopt.minimize(2 * y1 - 3 * y2 + x)
        sympyopt.add_constraint(ConstraintEq(2 * y1 + 3 * y2, 2), name="lin1")
        sympyopt.add_constraint(ConstraintEq(y1 ** 2 + 10.5 * y2, 1.1), name="lin23")

        omniqubo = Omniqubo(sympyopt)
        omniqubo.eq_to_obj(names="lin1", is_regexp=False, penalty=10)
        assert len(omniqubo.model.constraints) == 1
        omniqubo.eq_to_obj(names="lin23", is_regexp=False, penalty=2.0)
        assert len(omniqubo.model.constraints) == 0

        omniqubo = Omniqubo(sympyopt)
        assert len(omniqubo.model.constraints) == 2
        omniqubo.eq_to_obj(names="lin", penalty=1.0)
        assert len(omniqubo.model.constraints) == 2

        omniqubo.eq_to_obj(names="lin[0-9]+", penalty=1.0)
        assert len(omniqubo.model.constraints) == 0

    def test_qubo_isstatements(self):
        sopt = SympyOpt()
        x = sopt.bit_var("x")
        y = sopt.bit_var("y")
        z = sopt.bit_var("z")
        sopt.minimize(x ** 2 + (z + y) ** 2)
        omniqubo = Omniqubo(sopt)

        assert omniqubo.is_bm()
        assert omniqubo.is_qubo()
        assert omniqubo.is_hobo()
        assert omniqubo.is_pip()
        assert omniqubo.is_qip()
        assert not omniqubo.is_ilp()
        assert omniqubo.is_qcqp()
        assert not omniqubo.is_ising()

    def test_hobo_isstatements(self):
        sopt = SympyOpt()
        x = sopt.bit_var("x")
        y = sopt.bit_var("y")
        z = sopt.bit_var("z")
        sopt.minimize(x ** 2 + (z + y + x) ** 4)
        omniqubo = Omniqubo(sopt)

        assert omniqubo.is_bm()
        assert not omniqubo.is_qubo()
        assert omniqubo.is_hobo()
        assert omniqubo.is_pip()
        assert not omniqubo.is_qip()
        assert not omniqubo.is_ilp()
        assert not omniqubo.is_qcqp()
        assert not omniqubo.is_ising()
        assert not omniqubo.is_ising(locality=3)

    def test_ising_isstatements(self):
        sopt = SympyOpt()
        x = sopt.spin_var("x")
        y = sopt.spin_var("y")
        z = sopt.spin_var("z")
        sopt.minimize(x ** 2 + (z + y) ** 2)
        omniqubo = Omniqubo(sopt)

        assert omniqubo.is_bm()
        assert not omniqubo.is_qubo()
        assert not omniqubo.is_hobo()
        assert not omniqubo.is_pip()
        assert not omniqubo.is_qip()
        assert not omniqubo.is_ilp()
        assert not omniqubo.is_qcqp()
        assert omniqubo.is_ising()
        assert omniqubo.is_ising(locality=3)
        assert not omniqubo.is_ising(locality=1)

        sopt = SympyOpt()
        x = sopt.spin_var("x")
        y = sopt.spin_var("y")
        z = sopt.spin_var("z")
        sopt.minimize(x ** 2 + (z + y + x) ** 3)
        omniqubo = Omniqubo(sopt)

        assert omniqubo.is_bm()
        assert not omniqubo.is_qubo()
        assert not omniqubo.is_hobo()
        assert not omniqubo.is_pip()
        assert not omniqubo.is_qip()
        assert not omniqubo.is_ilp()
        assert not omniqubo.is_qcqp()
        assert not omniqubo.is_ising()
        assert omniqubo.is_ising(locality=3)
        assert not omniqubo.is_ising(locality=1)

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
        omniqubo = Omniqubo(sopt)

        assert not omniqubo.is_bm()
        assert not omniqubo.is_qubo()
        assert not omniqubo.is_hobo()
        assert omniqubo.is_pip()
        assert omniqubo.is_qip()
        assert omniqubo.is_ilp()
        assert omniqubo.is_qcqp()
        assert not omniqubo.is_ising()

        sopt.minimize(2 * x + 4 * y ** 2 - 3 * z ** 2)
        omniqubo = Omniqubo(sopt)

        assert not omniqubo.is_bm()
        assert not omniqubo.is_qubo()
        assert not omniqubo.is_hobo()
        assert omniqubo.is_pip()
        assert omniqubo.is_qip()
        assert not omniqubo.is_ilp()
        assert omniqubo.is_qcqp()
        assert not omniqubo.is_ising()

        c = ConstraintEq(2 * x - 3 * y ** 2 + 3.4 * z ** 2, 1.4 + 2 * x)
        sopt.add_constraint(c)
        omniqubo = Omniqubo(sopt)

        assert not omniqubo.is_bm()
        assert not omniqubo.is_qubo()
        assert not omniqubo.is_hobo()
        assert omniqubo.is_pip()
        assert not omniqubo.is_qip()
        assert not omniqubo.is_ilp()
        assert omniqubo.is_qcqp()
        assert not omniqubo.is_ising()

        sopt.minimize(2 * x + 4 * y ** 3 - 3 * z ** 2)
        omniqubo = Omniqubo(sopt)

        assert not omniqubo.is_bm()
        assert not omniqubo.is_qubo()
        assert not omniqubo.is_hobo()
        assert omniqubo.is_pip()
        assert not omniqubo.is_qip()
        assert not omniqubo.is_ilp()
        assert not omniqubo.is_qcqp()
        assert not omniqubo.is_ising()

    def test_nonpoly(self):
        sopt = SympyOpt()
        x = sopt.bit_var("x")
        sopt.minimize(sin(x))
        omniqubo = Omniqubo(sopt)

        assert omniqubo.is_bm()
        assert not omniqubo.is_qubo()
        assert not omniqubo.is_hobo()
        assert not omniqubo.is_pip()
        assert not omniqubo.is_qip()
        assert not omniqubo.is_ilp()
        assert not omniqubo.is_qcqp()
        assert not omniqubo.is_ising()

        sopt = SympyOpt()
        x = sopt.bit_var("x")
        sopt.minimize(x)
        sopt.add_constraint(ConstraintEq(2 * sin(x), 1))
        omniqubo = Omniqubo(sopt)

        assert omniqubo.is_bm()
        assert not omniqubo.is_qubo()
        assert not omniqubo.is_hobo()
        assert not omniqubo.is_pip()
        assert not omniqubo.is_qip()
        assert not omniqubo.is_ilp()
        assert not omniqubo.is_qcqp()
        assert not omniqubo.is_ising()

        sopt = SympyOpt()
        x = sopt.int_var("x")
        sopt.minimize(x)
        sopt.add_constraint(ConstraintEq(2 * sin(x), 1))
        omniqubo = Omniqubo(sopt)

        assert not omniqubo.is_bm()
        assert not omniqubo.is_qubo()
        assert not omniqubo.is_hobo()
        assert not omniqubo.is_pip()
        assert not omniqubo.is_qip()
        assert not omniqubo.is_ilp()
        assert not omniqubo.is_qcqp()
        assert not omniqubo.is_ising()
