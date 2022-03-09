from typing import Union

from qiskit.opflow import PauliSumOp
from qiskit_optimization import QuadraticProgram

from omniqubo.transpiler import TranspilerAbs

from ..sympyopt import SympyOpt


class SympyOptToQiskit(TranspilerAbs):
    """Transpile SympyOpt model into qiskit models

    At the moment mode can only be None, "qiskit_qp" or "qiskit_pso", first two
    returning qiskit_optimization.QuadraticProgram, and the last one returning
    qiskit.opflow.PauliSumOp. Transpiler assumes that for the first output model
    is a QIP, and for the last it is Ising model.

    :param mode: type of the model returned by transpile
    """

    def __init__(self, mode: str = None) -> None:
        if mode is None:
            mode = "qiskit_qp"
        assert mode == "qiskit_op" or mode == "qiskit_pso"
        self.mode = mode

    def transpile(self, model: SympyOpt) -> Union[QuadraticProgram, PauliSumOp]:
        raise NotImplementedError()

    def can_transpile(self, model: SympyOpt) -> bool:
        """Check if SympyOpt can be transpiled

        Currently equivalent to the fact that SympyOpt is QIP for qiskit_op, and
        Ising model for qiskit_pso.

        :param model: checked model
        :return: flag denoting if model can be transpiled
        """
        raise NotImplementedError()
