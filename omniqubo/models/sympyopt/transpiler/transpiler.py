from copy import deepcopy
from typing import Union

from docplex.mp.model import Model

from ..sympyopt import SympyOpt
from .docplex_to_sympyopt import DocplexToSympyopt


def transpile(model: Union[SympyOpt, Model]) -> SympyOpt:
    """Transpile optimization problem into SympyOpt model.

    Only accepts SympyOpt or Docplex model.

    :param model: model to be transpiled
    :raises ValueError: if the argument is of inappropriate type
    :return: transpiled model
    """
    if isinstance(model, SympyOpt):
        return deepcopy(model)
    elif isinstance(model, Model):
        return DocplexToSympyopt().transpile(model)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
