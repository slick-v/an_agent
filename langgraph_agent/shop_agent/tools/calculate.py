"""CALCULATE tools: exact maths (LLMs are unreliable at arithmetic)."""
import ast
import math
import operator

from langchain_core.tools import tool

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCTIONS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt}


def _evaluate(node):
    # walk the parsed expression and allow only numbers, maths operators and a few
    # functions — unlike eval(), nothing else can run
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):
        return _FUNCTIONS[node.func.id](*[_evaluate(a) for a in node.args])
    raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")


@tool
def calculator(expression: str):
    """Evaluate a maths expression exactly, e.g. '(3499 * 2) * 0.9' or 'round(1234.567, 2)'.
    Supports + - * / // % ** and round, abs, min, max, sqrt."""
    result = _evaluate(ast.parse(expression, mode="eval").body)
    return {"expression": expression, "result": result}
