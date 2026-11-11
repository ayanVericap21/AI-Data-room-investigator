"""calculate: deterministic arithmetic so the model never does maths itself.

Parses the expression with ``ast`` and evaluates only numbers and arithmetic operators, in
exact ``Decimal`` arithmetic. Names, calls, attributes and everything else are rejected.
"""
from __future__ import annotations

import ast
import operator
from decimal import Decimal

from app.tools.base import ToolContext, ToolError
from app.tools.schemas import CalculateInput, CalculateOutput

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}
_MAX_EXPONENT = 10


def _eval(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        number = Decimal(str(node.value))
        if not number.is_finite():
            raise ToolError("invalid_expression", "Numbers must be finite.")
        return number
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (right != right.to_integral_value() or abs(right) > _MAX_EXPONENT):
            raise ToolError("invalid_expression", f"Exponents must be whole numbers no larger than {_MAX_EXPONENT}.")
        return _BINARY[type(node.op)](left, right)
    raise ToolError("invalid_expression", "Only numbers, + - * / % ** and parentheses are allowed.")


def calculate(ctx: ToolContext, inp: CalculateInput) -> CalculateOutput:
    try:
        result = _eval(ast.parse(inp.expression.strip(), mode="eval"))
    except ToolError:
        raise
    except SyntaxError as exc:
        raise ToolError("invalid_expression", "Could not parse the expression. Use plain numbers and operators.") from exc
    except ArithmeticError as exc:  # division by zero, invalid operation, overflow
        raise ToolError("invalid_expression", f"Arithmetic error: {exc.__class__.__name__}.") from exc
    return CalculateOutput(expression=inp.expression.strip(), result=format(result.normalize(), "f"))
