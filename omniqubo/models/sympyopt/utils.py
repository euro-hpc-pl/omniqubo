from sympy import Expr, Float, preorder_traversal


def _approx_sympy_expr(expr: Expr) -> Expr:
    for a in preorder_traversal(expr):
        if isinstance(a, Float):
            expr = expr.xreplace({a: round(a, 15)})
    return expr
