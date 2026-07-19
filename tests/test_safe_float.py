"""Regression tests for ``server_simple._safe_float``.

These pin the behavior that a type-only fix (relaxing the static type so
``float`` accepts a persisted ``object``) must preserve: every
float-convertible input is still converted, and falsy inputs still coerce to
``0.0``. In particular a non-zero ``Decimal`` must not be flattened to ``0.0``
(the regression an ``isinstance`` guard would have introduced).
"""

from decimal import Decimal
from fractions import Fraction

import pytest

from claude_teams import server_simple


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("2.5"), 2.5),
        (Fraction(7, 2), 3.5),
        (3, 3.0),
        (1.5, 1.5),
        ("4.25", 4.25),
        (True, 1.0),
    ],
)
def test_safe_float_preserves_float_convertible_values(
    value: object, expected: float
) -> None:
    assert server_simple._safe_float(value) == expected


@pytest.mark.parametrize("value", [None, "", 0, 0.0, [], {}])
def test_safe_float_falsy_inputs_coerce_to_zero(value: object) -> None:
    assert server_simple._safe_float(value) == 0.0


def test_safe_float_non_numeric_string_returns_zero() -> None:
    assert server_simple._safe_float("not-a-number") == 0.0
