from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

MIN_SENSE = "min"
MAX_SENSE = "max"


class ModelAbs(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.variables = dict()  # type: Dict[str,Any]
        self.constraints = dict()  # type: Dict[str,Any]
        self.sense = MIN_SENSE
        self.objective = None

        pass

    @abstractmethod
    def minimize(self, obj) -> None:
        """Set the function to be minimized.

        :param obj: minimized expression
        """
        pass

    @abstractmethod
    def maximize(self, obj) -> None:
        """Set the function to be maximized.

        :param obj: maximized expression
        """
        pass

    @abstractmethod
    def add_constraint(self, constraint, name: str = None) -> None:
        """Add constraint to the model.

        If name is not provided, a random string is generated.

        :param constraint: The constraint
        :param name: name of the constraint, defaults to random name
        :raises ValueError: if variables are not present in the model
        """
        pass

    @abstractmethod
    def list_constraints(self) -> Dict:
        """Return the dictionary of the constraints.

        :return: Dictionary of the constraints.
        """
        pass

    @abstractmethod
    def get_objective(self):
        """Return the objective function.

        :return: The objective function
        """
        pass

    @abstractmethod
    def get_constraint(self, name: str):
        """Return the constraint of the given name.

        :param name: name of the constraint
        :return: the constraint
        """
        pass

    @abstractmethod
    def get_var(self, name: str):
        """Return the variable of the given name.

        :param name: name of the variable
        :return: the variable
        """
        pass

    @abstractmethod
    def get_vars(self) -> Dict:
        """Return the dictionary of variables.

        :return: dictionary of variables
        """
        pass

    @abstractmethod
    def int_var(self, name: str, lb: int = None, ub: int = None):
        """Create and return integer variable.

        :param name: name of the variable
        :param lb: minimal value, defaults to -INF
        :param ub: maximal value, defaults to INF
        :raises ValueError: if the name is already used
        :return: the Sympy variable
        """
        pass

    @abstractmethod
    def real_var(self, name: str, lb: float = None, ub: float = None):
        """Create and return real variable.

        :param name: name of the variable
        :param lb: minimal value, defaults to -INF
        :param ub: maximal value, defaults to INF
        :raises ValueError: if the name is already used
        :return: the Sympy variable
        """
        pass

    @abstractmethod
    def bit_var(self, name: str):
        """Create and return binary variable.

        :param name: name of the variable
        :raises ValueError: if the name is already used
        :return: the Sympy variable
        """
        pass

    @abstractmethod
    def spin_var(self, name: str):
        """Create and return spin variable

        :param name: name of the variable
        :raises ValueError: if the name is already used
        :return: the Sympy variable
        """
        pass

    @abstractmethod
    def is_lip(self) -> bool:
        """Check if model is Linear Integer Program.

        Model is Linear Integer Program if all variables are bits or integers,
        and objective and constraints are linear

        :return: flag stating if the model is Linear Integer Program
        """
        pass

    @abstractmethod
    def is_qip(self) -> bool:
        """Check if model is Quadratic Integer Program.

        Model is Quadratic Integer Program if all variables are bits or
        integers, objective is quadratic polynomial, and constraints are linear.

        :return: flag stating if the model is Quadratic Integer Program
        """
        pass

    @abstractmethod
    def is_pp(self) -> bool:
        """Check if model is Polynomial Integer Program.

        Model is Polynomial Integer Program if all variables are bits or
        integers, objective and constraints are polynomials.

        :return: flag stating if the model is Polynomial Integer Program
        """
        pass

    @abstractmethod
    def is_qcqp(self) -> bool:
        """Check if model is Quadratically Constrained Quadratic Program.

        Model is Quadratically Constrained Quadratic Program if all variables
        are bits or integers, objective and constraints are quadratic
        polynomials.

        :return: flag stating if the model is Quadratically Constrained
            Quadratic Program
        """
        pass

    @abstractmethod
    def is_bm(self) -> bool:
        """Check if model is Binary Model.

        Model is Binary Model if all variables are bits or spins.

        :return: flag stating if the model is Binary Model
        """
        pass

    @abstractmethod
    def is_qubo(self) -> bool:
        """Check if model is Quadratic Unconstrained Binary Optimization.

        Model is Quadratic Unconstrained Binary Optimization if all variables
        are bits, objective function is quadratic polynomial and there are no
        constraints.

        :return: flag stating if the model is Quadratic Unconstrained Binary
            Optimization
        """
        pass

    @abstractmethod
    def is_ising(self, locality: int = None) -> bool:
        """Check if model is an Ising Model.

        Model is Ising model if all variables are spins, objective function is
        a polynomial of at most locality order and there are no constraints.

        :param locality: maximal locality, defaults to 2
        :return: flag stating if the model is Quadratic Unconstrained Binary
            Optimization
        """
        pass

    @abstractmethod
    def is_hobo(self) -> bool:
        """Check if model is Higher Order Binary Optimization.

        Model is Higher Order Binary Optimization if all variables are bits,
        objective function is a polynomial and there are no constraints.

        :return: flag stating if the model is Higher Order Binary Optimization
        """
        pass
