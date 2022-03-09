from pulp import LpProblem

from omniqubo.transpiler import TranspilerAbs

from ..sympyopt import SympyOpt


class PulpToSympyopt(TranspilerAbs):
    """Transpiler for transforming PuLP LpProblem into SymptOpt model

    Transpiler can transform any LpProblem.
    """

    def transpile(self, model: LpProblem) -> SympyOpt:
        """Transpile LpProblem into SympyOpt model

        :param model: model to be transpiled
        :return: equivalent SympyOpt model
        """
        raise NotImplementedError()

    def can_transpile(self, _: LpProblem) -> bool:
        """Check if model can be transpiled

        Currently all LpProblem can be transpiled.

        :type model: model to be transpiled
        :return: flag denoting if model can be transpiled
        """
        return True
