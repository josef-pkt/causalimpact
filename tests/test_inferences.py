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

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.arima_process import ArmaProcess

from causalimpact import CausalImpact
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


def test_default_causal_cto_w_positive_signal():
    np.random.seed(1)
    ar = np.r_[1, 0.9]
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)
    X = 100 + arma_process.generate_sample(nsample=100)
    y = 1.2 * X + np.random.normal(size=(100))
    y[70:] += 1
    data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])
    ci = CausalImpact(data, [0, 69], [70, 99])
    assert ci.p_value < 0.05


def test_causal_cto_w_positive_signal_no_standardization():
    np.random.seed(1)
    ar = np.r_[1, 0.9]
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)
    X = 100 + arma_process.generate_sample(nsample=100)
    y = 1.2 * X + np.random.normal(size=(100))
    y[70:] += 1
    data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])
    ci = CausalImpact(data, [0, 69], [70, 99], standardize=False)
    assert ci.p_value < 0.05


def test_default_causal_cto_w_negative_signal():
    np.random.seed(1)
    ar = np.r_[1, 0.9]
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)
    X = 100 + arma_process.generate_sample(nsample=100)
    y = 1.2 * X + np.random.normal(size=(100))
    y[70:] -= 1
    data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])
    ci = CausalImpact(data, [0, 69], [70, 99])
    assert ci.p_value < 0.05


def test_causal_cto_w_negative_signal_no_standardization():
    np.random.seed(1)
    ar = np.r_[1, 0.9]
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)
    X = 100 + arma_process.generate_sample(nsample=100)
    y = 1.2 * X + np.random.normal(size=(100))
    y[70:] -= 1
    data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])
    ci = CausalImpact(data, [0, 69], [70, 99], standardize=False)
    assert ci.p_value < 0.05


def test_causal_cto_w_negative_signal_no_standardization():
    np.random.seed(1)
    ar = np.r_[1, 0.9]
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)
    X = 100 + arma_process.generate_sample(nsample=100)
    y = 1.2 * X + np.random.normal(size=(100))
    y[70:] -= 1
    data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])
    ci = CausalImpact(data, [0, 69], [70, 99], standardize=False)
    assert ci.p_value < 0.05


def test_default_causal_cto_no_signal():
    np.random.seed(1)
    ar = np.r_[1, 0.9]
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)
    X = 100 + arma_process.generate_sample(nsample=100)
    y = 1.2 * X + np.random.normal(size=(100))
    data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])
    ci = CausalImpact(data, [0, 69], [70, 99])
    assert ci.p_value > 0.05


def test_lower_upper_percentile():
    np.random.seed(1)
    ar = np.r_[1, 0.9]
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)
    X = 100 + arma_process.generate_sample(nsample=100)
    y = 1.2 * X + np.random.normal(size=(100))
    data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])
    ci = CausalImpact(data, [0, 69], [70, 99])
    ci.lower_upper_percentile == [2.5, 97.5]
