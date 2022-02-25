from sympy import Expr, Float, preorder_traversal


# approximates the values of float numbers in sympy expression
def _approx_sympy_expr(expr: Expr) -> Expr:
    dict_rule = {}
    for a in preorder_traversal(expr):
        if isinstance(a, Float):
            dict_rule[a] = round(a, 15)
    expr = expr.xreplace(dict_rule)
    return expr
