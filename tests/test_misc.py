# MIT License
#
# Copyright (c) 2018 Dafiti OpenSource
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests for module misc.py"""

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_almost_equal

from causalimpact.misc import get_z_score, standardize, unstandardize


def test_basic_standardize():
    data = {
        'c1': [1, 4, 8, 9, 10],
        'c2': [4, 8, 12, 16, 20]
    }
    data = pd.DataFrame(data)

    result, (mu, sig) = standardize(data)
    assert_almost_equal(
        np.zeros(data.shape[1]),
        result.mean().values
    )

    assert_almost_equal(
        np.ones(data.shape[1]),
        result.std(ddof=0).values
    )

def test_standardize_w_various_distinct_inputs():
    test_data = [[1, 2, 1], [1, np.nan, 3], [10, 20, 30]]
    test_data = [pd.DataFrame(data, dtype="float") for data in test_data]
    for data in test_data:
        result, (mu, sig) = standardize(data)
        pd.util.testing.assert_frame_equal(unstandardize(result, (mu, sig)), data)

def test_standardize_raises_single_input():
    with pytest.raises(ValueError):
        standardize(pd.DataFrame([1]))


def test_get_z_score():
    assert get_z_score(0.5) == 0.
    assert round(get_z_score(0.9177), 2) == 1.39
