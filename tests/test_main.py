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

"""Tests for module main.py"""


import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_less
import pandas as pd
from pandas.util.testing import assert_frame_equal
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.structural import UnobservedComponentsResultsWrapper
from causalimpact import CausalImpact
from causalimpact.misc import standardize


@pytest.fixture
def rand_data():
    return pd.DataFrame(np.random.randn(200, 3), columns=["y", "x1", "x2"])


@pytest.fixture
def pre_int_period():
    return [0, 100]


@pytest.fixture
def post_int_period():
    return [100, 200]


def test_default_causal_cto(rand_data, pre_int_period, post_int_period):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    assert_frame_equal(ci.data, rand_data)
    assert ci.pre_period == pre_int_period
    assert ci.post_period == post_int_period
    pre_data = rand_data.iloc[pre_int_period[0]: pre_int_period[1], :]
    assert_frame_equal(ci.pre_data, pre_data)

    post_data = rand_data.iloc[post_int_period[0]: post_int_period[1], :]
    assert_frame_equal(ci.post_data, post_data)

    assert ci.alpha == 0.05
    normed_pre_data, (mu, sig) = standardize(pre_data)
    assert_frame_equal(ci.normed_pre_data, normed_pre_data)

    normed_post_data = (post_data - mu) / sig
    assert_frame_equal(ci.normed_post_data, normed_post_data)

    assert ci.mu_sig == (mu[0], sig[0])
    assert ci.model_args == {'standardize': True}

    assert isinstance(ci.model, UnobservedComponents)
    assert_array_equal(ci.model.endog, normed_pre_data.iloc[:, 0].values.reshape(-1, 1))
    assert_array_equal(ci.model.exog, normed_pre_data.iloc[:, 1:].values.reshape(
            -1,
            rand_data.shape[1] - 1
        )
    )
    assert ci.model.endog_names == 'y'
    assert ci.model.exog_names == ['x1', 'x2']
    assert ci.model.k_endog == 1
    assert ci.model.level
    assert ci.model.trend_specification == 'local level'

    assert isinstance(ci.trained_model, UnobservedComponentsResultsWrapper)
    assert ci.trained_model.nobs == len(pre_data)

    assert ci.inferences is not None
    assert ci.p_value > 0 and ci.p_value < 1
    assert ci.n_sims == 1000


def test_causal_cto_w_no_standardization(rand_data, pre_int_period, post_int_period):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period, standardize=False)
    pre_data = rand_data.iloc[pre_int_period[0]: pre_int_period[1], :]
    post_data = rand_data.iloc[post_int_period[0]: post_int_period[1], :]
    assert ci.normed_pre_data is None
    assert ci.normed_post_data is None
    assert ci.mu_sig is None
    assert_array_equal(ci.model.endog, pre_data.iloc[:, 0].values.reshape(-1, 1))
    assert_array_equal(ci.model.exog, pre_data.iloc[:, 1:].values.reshape(
            -1,
            rand_data.shape[1] - 1
        )
    )
    assert ci.p_value > 0 and ci.p_value < 1


def test_causal_cto_w_custom_model(rand_data, pre_int_period, post_int_period):
    pre_data = rand_data.iloc[pre_int_period[0]: pre_int_period[1], :]
    post_data = rand_data.iloc[post_int_period[0]: post_int_period[1], :]
    model = UnobservedComponents(endog=pre_data.iloc[:, 0], level='llevel',
                                 exog=pre_data.iloc[:, 1:])

    ci = CausalImpact(rand_data, pre_int_period, post_int_period, model=model)

    assert ci.model.endog_names == 'y'
    assert ci.model.exog_names == ['x1', 'x2']
    assert ci.model.k_endog == 1
    assert ci.model.level
    assert ci.model.trend_specification == 'local level'

    assert isinstance(ci.trained_model, UnobservedComponentsResultsWrapper)
    assert ci.trained_model.nobs == len(pre_data)


def test_causal_cto_raises_on_None_input(rand_data, pre_int_period, post_int_period):
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(None, pre_int_period, post_int_period)
    assert str(excinfo.value) == 'data input cannot be empty'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, None, post_int_period)
    assert str(excinfo.value) == 'pre_period input cannot be empty'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, pre_int_period, None)
    assert str(excinfo.value) == 'post_period input cannot be empty'



