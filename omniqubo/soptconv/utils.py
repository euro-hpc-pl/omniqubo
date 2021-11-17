from copy import deepcopy

from docplex.mp.model import Model

from ..sympyopt import SympyOpt
from .docplex_to_sympyopt import DocplexToSymopt


def convert_to_sympyopt(model) -> SympyOpt:
    if isinstance(model, SympyOpt):
        return deepcopy(model)
    elif isinstance(model, Model):
        return DocplexToSymopt().convert(model)
    else:
        raise ValueError(f"Unknown model type {type(model)}. Please provide SympyOpt.")
