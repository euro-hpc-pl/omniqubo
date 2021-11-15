from ..sympyopt import SympyOpt


class ConvertToSymoptAbs:
    def __init__(self) -> None:
        pass

    def convert(self, model) -> SympyOpt:
        raise NotImplementedError

    def can_convert(self, model) -> bool:
        raise NotImplementedError
