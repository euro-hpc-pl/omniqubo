from typing import Union

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel

from omniqubo.transpiler import TranspilerAbs

from ..sympyopt import SympyOpt


class DimodToSympyopt(TranspilerAbs):
    """Transpiler for transforming dimod models into SymptOpt model

    Transpiler can transform any BinaryQuadraticModel and
    ConstrainedQuadraticModel.
    """

    def transpile(self, model: Union[BinaryQuadraticModel, ConstrainedQuadraticModel]) -> SympyOpt:
        """Transpile dimod model into SympyOpt model

        :param model: model to be transpiled
        :return: equivalent SympyOpt model
        """
        raise NotImplementedError()

    def can_transpile(self, _: Union[BinaryQuadraticModel, ConstrainedQuadraticModel]) -> bool:
        """Check if model can be transpiled

        Transpiler can transform any BinaryQuadraticModel and
        ConstrainedQuadraticModel.

        :type model: model to be transpiled
        :return: flag denoting if model can be transpiled
        """
        return True
