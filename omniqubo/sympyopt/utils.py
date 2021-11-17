import random
import string

from sympy import Expr, Float, preorder_traversal

RAND_STR_LEN = 16


def gen_random_str(n: int = None) -> str:
    if n is None:
        n = RAND_STR_LEN
    str_letters = string.ascii_uppercase + string.digits
    return "".join(random.choices(str_letters, k=n))


def _approx_sympy_expr(expr: Expr) -> Expr:
    for a in preorder_traversal(expr):
        if isinstance(a, Float):
            expr = expr.subs(a, round(a, 15))
    return expr
