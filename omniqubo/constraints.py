from abc import ABC, abstractmethod
from typing import Iterable, List

from omniqubo.vars import VarAbs


class ConstraintAbs(ABC):
    """Abstract class for constraints"""

    @abstractmethod
    def __init__(self) -> None:
        # important for example to avoid checking equalities coming from equality
        self.check_interpret = True
        pass

    @abstractmethod
    def is_eq_constraint(self) -> bool:
        """Check if the constraint is an equality

        Needs to be implemented
        :return: the flag which states if constraints is an equality
        """
        pass

    @abstractmethod
    def is_ineq_constraint(self) -> bool:
        """Check if the constraint is an inequality

        Needs to be implemented
        :return: the flag which states if constraints is an inequality
        """
        pass

    @abstractmethod
    def _list_unknown_vars(self, vars: Iterable[str]) -> List[VarAbs]:
        """Filters the list of variables into those NOT present in the constraint

        :param vars: list of variables to be filtered
        :return: filtered List of variables
        """
        pass
