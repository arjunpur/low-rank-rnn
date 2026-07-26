"""Runtime shape checking shared by public numerical APIs."""

from typing import Any, get_type_hints

from beartype import beartype
from jaxtyping import jaxtyped


def _typechecked(function: Any) -> Any:
    """Apply jaxtyping after resolving annotations transformed by Trickle."""
    wrapped = function
    while True:
        wrapped.__annotations__ = get_type_hints(wrapped, include_extras=True)
        if not hasattr(wrapped, "__wrapped__"):
            break
        wrapped = wrapped.__wrapped__
    return jaxtyped(typechecker=beartype)(function)


typechecked = _typechecked
