import numpy as np
import pytest
from numpy.testing import assert_equal

from corrscope.utils.windows import leftpad, midpad


def test_leftpad():
    before = 10
    data = np.linspace(before - 1, 0, before)

    after = 3
    assert_equal(leftpad(data, after), data[-after:])

    after = 10
    assert_equal(leftpad(data, after), data)

    after = 25
    lp = leftpad(data, after)
    assert_equal(lp[-before:], data)
    assert_equal(lp[:-before], 0)


@pytest.mark.parametrize("before", [10, 11])
def test_midpad(before):
    # Midpad should work for both odd and even arrays.
    data = np.arange(before)

    # Test shrinking.
    after = before - 6
    half = (before - after) // 2
    assert_equal(midpad(data, after), data[half:-half])

    # Test identity.
    after = before
    assert_equal(midpad(data, after), data)

    # Test expanding.
    after = before + 12
    half = (after - before) // 2
    lp = midpad(data, after)

    assert_equal(lp[:half], 0)
    assert_equal(lp[half:-half], data)
    assert_equal(lp[-half:], 0)

    # I don't test odd values, except to ensure sizes are correct.
    for after in [before - 5, before + 15]:
        assert len(midpad(data, after)) == after
