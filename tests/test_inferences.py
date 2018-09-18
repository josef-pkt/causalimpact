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

import pytest
import pandas as pd

from causalimpact.inferences import Inferences


@pytest.fixture
def inferer():
    return Inferences(10)


def test_inferer_cto():
    inferer = Inferences(10)
    assert inferer.n_sims == 10
    assert inferer.inferences is None
    assert inferer.p_value is None


def test_p_value_read_only(inferer):
    with pytest.raises(AttributeError):
        inferer.p_value = 0.4
        inferer.p_value = 0.3


def test_p_value_bigger_than_one(inferer):
    with pytest.raises(ValueError):
        inferer.p_value = 2


def test_p_value_lower_than_zero(inferer):
    with pytest.raises(ValueError):
        inferer.p_value = -1


def test_inferences_read_only(inferer):
    with pytest.raises(AttributeError):
        inferer.inferences = pd.DataFrame([1, 2, 3])
        inferer.inferences = pd.DataFrame([1, 2, 3])


def test_inferences_raises_invalid_input(inferer):
    with pytest.raises(ValueError):
        inferer.inferences = 1


def test_summarization_raises(inferer):
    with pytest.raises(RuntimeError):
        inferer.summarize_posterior_inferences()
