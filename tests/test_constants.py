"""Tests for brahim_energy.constants."""

import math

from brahim_energy.constants import (
    BETA_SECURITY,
    BRAHIM_CENTER,
    BRAHIM_SEQUENCE,
    BRAHIM_SUM,
    D,
    DIMENSION_NAMES,
    GAMMA,
    GENESIS_CONSTANT,
    LUCAS_NUMBERS,
    PHI,
    REGULARITY_THRESHOLD,
    TOTAL_STATES,
    x_from_D,
)


def test_phi_value():
    assert abs(PHI - 1.6180339887498949) < 1e-12


def test_phi_identity():
    # PHI satisfies PHI^2 = PHI + 1
    assert abs(PHI ** 2 - PHI - 1) < 1e-12


def test_genesis_constant():
    assert abs(GENESIS_CONSTANT - 2 / 901) < 1e-15


def test_beta_security():
    assert abs(BETA_SECURITY - (math.sqrt(5) - 2)) < 1e-12


def test_gamma():
    assert abs(GAMMA - 1 / PHI ** 4) < 1e-12


def test_brahim_sequence_length():
    assert len(BRAHIM_SEQUENCE) == 10


def test_brahim_sum():
    # Not the sum of the sequence — it's the mirror constant 214
    assert BRAHIM_SUM == 214


def test_brahim_center():
    assert BRAHIM_CENTER == 107
    assert BRAHIM_CENTER == BRAHIM_SUM // 2


def test_lucas_numbers():
    assert LUCAS_NUMBERS == [1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]
    # Lucas recurrence: L(n) = L(n-1) + L(n-2)
    for i in range(2, len(LUCAS_NUMBERS)):
        assert LUCAS_NUMBERS[i] == LUCAS_NUMBERS[i - 1] + LUCAS_NUMBERS[i - 2]


def test_total_states():
    assert TOTAL_STATES == sum(LUCAS_NUMBERS)
    assert TOTAL_STATES == 840


def test_dimension_names():
    assert len(DIMENSION_NAMES) == 12
    assert DIMENSION_NAMES[0] == "PERCEPTION"
    assert DIMENSION_NAMES[11] == "UNIFICATION"


def test_d_function():
    # D(1/PHI) should be 1.0
    assert abs(D(1 / PHI) - 1.0) < 1e-10
    # D(1/PHI^2) should be 2.0
    assert abs(D(1 / PHI ** 2) - 2.0) < 1e-10


def test_d_inverse():
    for d_val in [1.0, 2.5, 5.0, 10.0]:
        x = x_from_D(d_val)
        assert abs(D(x) - d_val) < 1e-10


def test_d_raises_on_zero():
    import pytest
    with pytest.raises(ValueError):
        D(0)


def test_x_from_d():
    assert abs(x_from_D(0) - 1.0) < 1e-10
    assert abs(x_from_D(1) - 1 / PHI) < 1e-10
