from sympyopt import SympyOpt
from copy import deepcopy

def convert_to_sympyopt(model):
    if isinstance(model, SympyOpt):
        return deepcopy(model)