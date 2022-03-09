from copy import deepcopy
from typing import Union

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel
from docplex.mp.model import Model
from pulp import LpProblem
from qiskit.opflow import PauliSumOp
from qiskit_optimization import QuadraticProgram

from ..sympyopt import SympyOpt
from .dimod_to_sympyopt import DimodToSympyopt
from .docplex_to_sympyopt import DocplexToSympyopt
from .pulp_to_sympyopt import PulpToSympyopt
from .qiskit_to_sympyopt import QiskitToSympyopt


def transpile(model: Union[SympyOpt, Model]) -> SympyOpt:
    """Transpile optimization problem into SympyOpt model

    Only accepts SympyOpt or Docplex model.

    :param model: model to be transpiled
    :raises ValueError: if the argument is of inappropriate type
    :return: transpiled model
    """
    if isinstance(model, SympyOpt):
        return deepcopy(model)
    elif isinstance(model, Model):
        return DocplexToSympyopt().transpile(model)
    elif isinstance(model, LpProblem):
        return PulpToSympyopt().transpile(model)
    elif isinstance(model, (QuadraticProgram, PauliSumOp)):
        return QiskitToSympyopt().transpile(model)
    elif isinstance(model, (BinaryQuadraticModel, ConstrainedQuadraticModel)):
        return DimodToSympyopt().transpile(model)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
