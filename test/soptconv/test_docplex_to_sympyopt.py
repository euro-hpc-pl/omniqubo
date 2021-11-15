from omniqubo.soptconv.docplex_to_sympyopt import DocplexToSymopt
from omniqubo.sympyopt.sympyopt import SympyOpt
from docplex.mp.model import Model

class TestDocplexToSymopt:
    def test_const_objectives(self):
        mdl = Model(name="tsp")
        sympymodel = DocplexToSymopt().convert(mdl)
        assert sympymodel == SympyOpt()

    def test_const_objectives(self):
        mdl = Model(name="tsp")
        mdl.minimize(1)
        sympymodel = DocplexToSymopt().convert(mdl)
        sympyopt = SympyOpt()
        sympyopt.minimize(1)
        assert sympymodel == sympyopt

    def test_max_const_objectives(self):
        mdl = Model(name="tsp")
        mdl.maximize(1)
        sympymodel = DocplexToSymopt().convert(mdl)
        sympyopt = SympyOpt()
        sympyopt.maximize(1)
        assert sympymodel == sympyopt

