from typing import Union

from qiskit.opflow import PauliSumOp
from qiskit_optimization import QuadraticProgram

from omniqubo.transpiler import TranspilerAbs

from ..sympyopt import SympyOpt


class QiskitToSympyopt(TranspilerAbs):
    """Transpiler for transforming qiskit models into SymptOpt model

    Transpiler can transform any QuadraticProgram, and PauliSumOp consisting
    of Z and I Pauli terms only.
    """

    def transpile(self, model: Union[QuadraticProgram, PauliSumOp]) -> SympyOpt:
        """Transpile qiskit models into SympyOpt model

        :param model: model to be transpiled
        :return: equivalent SympyOpt model
        """
        raise NotImplementedError()

    def can_transpile(self, model: Union[QuadraticProgram, PauliSumOp]) -> bool:
        """Check if model can be transpiled

        Currently all QuadraticProgram can be transpiled. PauliSumOp can be
        transpiled if it consists of I and Z Pauli terms only.

        :type model: model to be transpiled
        :return: flag denoting if model can be transpiled
        """
        if isinstance(model, QuadraticProgram):
            return True
        else:  # PauliSumOp
            raise NotImplementedError()
