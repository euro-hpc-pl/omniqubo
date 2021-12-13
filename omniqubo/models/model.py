from __future__ import annotations

from abc import ABC, abstractmethod


class ModelAbs(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def is_qubo(self) -> bool:
        pass

    @abstractmethod
    def is_ising(self, locality: int = None) -> bool:
        pass
