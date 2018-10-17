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

"""
Tests for module main.py. Fixtures comes from file conftest.py located at the same dir
of this file.
"""

import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.core.indexes.range import RangeIndex
from pandas.util.testing import assert_frame_equal, assert_series_equal
from statsmodels.tsa.statespace.structural import (UnobservedComponents,
                                                   UnobservedComponentsResultsWrapper)

from causalimpact import CausalImpact
from causalimpact.misc import standardize


def test_default_causal_cto(rand_data, pre_int_period, post_int_period):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    assert_frame_equal(ci.data, rand_data)
    assert ci.pre_period == pre_int_period
    assert ci.post_period == post_int_period
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
    assert_frame_equal(ci.pre_data, pre_data)

    post_data = rand_data.loc[post_int_period[0]: post_int_period[1], :]
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


def test_default_causal_cto_w_date(date_rand_data, pre_str_period, post_str_period):
    ci = CausalImpact(date_rand_data, pre_str_period, post_str_period)
    assert_frame_equal(ci.data, date_rand_data)
    assert ci.pre_period == pre_str_period
    assert ci.post_period == post_str_period
    pre_data = date_rand_data.loc[pre_str_period[0]: pre_str_period[1], :]
    assert_frame_equal(ci.pre_data, pre_data)

    post_data = date_rand_data.loc[post_str_period[0]: post_str_period[1], :]
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
            date_rand_data.shape[1] - 1
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


def test_default_causal_cto_no_exog(rand_data, pre_int_period, post_int_period):
    rand_data = pd.DataFrame(rand_data.iloc[:, 0])
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    assert_frame_equal(ci.data, rand_data)
    assert ci.pre_period == pre_int_period
    assert ci.post_period == post_int_period
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
    assert_frame_equal(ci.pre_data, pre_data)

    post_data = rand_data.loc[post_int_period[0]: post_int_period[1], :]
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
    assert ci.model.exog is None
    assert ci.model.endog_names == 'y'
    assert ci.model.exog_names is None
    assert ci.model.k_endog == 1
    assert ci.model.level
    assert ci.model.trend_specification == 'local level'

    assert isinstance(ci.trained_model, UnobservedComponentsResultsWrapper)
    assert ci.trained_model.nobs == len(pre_data)

    assert ci.inferences is not None
    assert ci.p_value > 0 and ci.p_value < 1
    assert ci.n_sims == 1000


def test_default_causal_cto_w_np_array(rand_data, pre_int_period, post_int_period):
    data = rand_data.values
    ci = CausalImpact(data, pre_int_period, post_int_period)
    assert_array_equal(ci.data, data)
    assert ci.pre_period == pre_int_period
    assert ci.post_period == post_int_period
    pre_data = pd.DataFrame(data[pre_int_period[0]: pre_int_period[1] + 1, :])
    assert_frame_equal(ci.pre_data, pre_data)

    post_data = pd.DataFrame(data[post_int_period[0]: post_int_period[1] + 1, :])
    post_data.index = RangeIndex(start=len(pre_data), stop=len(rand_data))
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
            data.shape[1] - 1
        )
    )
    assert ci.model.endog_names == 'y'
    assert ci.model.exog_names == [1, 2]
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
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
    post_data = rand_data.loc[post_int_period[0]: post_int_period[1], :]
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
    pre_data = rand_data.loc[pre_int_period[0]: pre_int_period[1], :]
    post_data = rand_data.loc[post_int_period[0]: post_int_period[1], :]
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


def test_invalid_data_input_raises():
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact('test', [0, 5], [5, 10])
    assert str(excinfo.value) == 'Could not transform input data to pandas DataFrame.'

    data = [1, 2, 3, 4, 5, 6, 2 + 1j]
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(data, [0, 3], [3, 6])
    assert str(excinfo.value) == 'Input data must contain only numeric values.'

    data = np.random.randn(10, 2)
    data[0, 1] = np.nan
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(data, [0, 3], [3, 6])
    assert str(excinfo.value) == 'Input data cannot have NAN values.'


def test_invalid_response_raises():
    data = np.random.rand(100, 2)
    data[:, 0] = np.ones(len(data)) * np.nan
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(data, [0, 50], [50, 100])
    assert str(excinfo.value) == 'Input response cannot have just Null values.'

    data[0:2, 0] = 1    
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(data, [0, 50], [50, 100])
    assert str(excinfo.value) == ('Input response must have more than 3 non-null points '
        'at least.')

    data[0:3, 0] = 1    
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(data, [0, 50], [50, 100])
    assert str(excinfo.value) == 'Input response cannot be constant.'


def test_invalid_alpha_raises(rand_data, pre_int_period, post_int_period):
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, pre_int_period, post_int_period, alpha=1)
    assert str(excinfo.value) == 'alpha must be of type float.'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, pre_int_period, post_int_period, alpha=2.)
    assert str(excinfo.value) == (
        'alpha must range between 0 (zero) and 1 (one) inclusive.')


def test_custom_model_input_validation(rand_data, pre_int_period, post_int_period):
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, pre_int_period, post_int_period, model='test')
    assert str(excinfo.value) == 'Input model must be of type UnobservedComponents.'

    ucm = UnobservedComponents(rand_data.iloc[:101, 0], level='llevel',
        exog=rand_data.iloc[:101, 1:])
    ucm.level = False
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, pre_int_period, post_int_period, model=ucm)
    assert str(excinfo.value) == 'Model must have level attribute set.'

    ucm = UnobservedComponents(rand_data.iloc[:101, 0], level='llevel',
        exog=rand_data.iloc[:101, 1:])
    ucm.exog = None
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, pre_int_period, post_int_period, model=ucm)
    assert str(excinfo.value) == 'Model must have exog attribute set.'

    ucm = UnobservedComponents(rand_data.iloc[:101, 0], level='llevel',
        exog=rand_data.iloc[:101, 1:])
    ucm.data = None
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, pre_int_period, post_int_period, model=ucm)
    assert str(excinfo.value) == 'Model must have data attribute set.'


def test_kwargs_validation(rand_data, pre_int_period, post_int_period):
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, pre_int_period, post_int_period,
                          standardize='yes')
    assert str(excinfo.value) == 'Standardize argument must be of type bool.'


def test_periods_validation(rand_data, date_rand_data):
    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, [5, 10], [4, 7])
    assert str(excinfo.value) == ('Values in training data cannot be present in the '
        'post-intervention data. Please fix your pre_period value to cover at most one '
        'point less from when the intervention happened.')

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(date_rand_data, ['20180101', '20180201'],
                          ['20180110', '20180210'])
    assert str(excinfo.value) == ('Values in training data cannot be present in the '
        'post-intervention data. Please fix your pre_period value to cover at most one '
        'point less from when the intervention happened.')

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, [5, 10], [15, 11])
    assert str(excinfo.value) == 'post_period last number must be bigger than its first.'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(date_rand_data, ['20180101', '20180110'],
                          ['20180115', '20180111'])
    assert str(excinfo.value) == 'post_period last number must be bigger than its first.'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, [0, 2], [15, 11])
    assert str(excinfo.value) == 'pre_period must span at least 3 time points.'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(date_rand_data, ['20180101', '20180102'],
                          ['20180115', '20180111'])
    assert str(excinfo.value) == 'pre_period must span at least 3 time points.'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, [5, 0], [15, 11])
    assert str(excinfo.value) == 'pre_period last number must be bigger than its first.'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(date_rand_data, ['20180105', '20180101'],
                          ['20180115', '20180111'])
    assert str(excinfo.value) == 'pre_period last number must be bigger than its first.'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, 0, [15, 11])
    assert str(excinfo.value) == 'Input period must be of type list.'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(date_rand_data, '20180101', ['20180115', '20180130'])
    assert str(excinfo.value) == 'Input period must be of type list.'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, [0, 10, 30], [15, 11])
    assert str(excinfo.value) == ('Period must have two values regarding the beginning '
        'and end of the pre and post intervention data.')

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, [0, None], [15, 11])
    assert str(excinfo.value) == 'Input period cannot have `None` values.'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, [0, 5.5], [15, 11])
    assert str(excinfo.value) == 'Input must contain either int or str.'

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, [-2, 10], [11, 20])
    assert str(excinfo.value) == (
        '-2 not present in input data index.'
    )

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, [0, 10], [11, 2000])
    assert str(excinfo.value) == (
        '2000 not present in input data index.'
    )

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(rand_data, ['20180101', '20180110'],
                          ['20180111', '20180130'])
    assert str(excinfo.value) == (
        '20180101 not present in input data index.'
    )

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(date_rand_data, ['20180101', '20180110'],
                          ['20180111', '20200130'])
    assert str(excinfo.value) == ('20200130 not present in input data index.')

    with pytest.raises(ValueError) as excinfo:
        ci = CausalImpact(date_rand_data, ['20170101', '20180110'],
                          ['20180111', '20180120'])
    assert str(excinfo.value) == ('20170101 not present in input data index.')


def test_default_causal_inferences(fix_path):
    np.random.seed(1)
    data = pd.read_csv(os.path.join(fix_path, 'google_data.csv'))
    del data['t']

    pre_period = [0, 60]
    post_period = [61, 90]

    ci = CausalImpact(data, pre_period, post_period)
    assert int(ci.summary_data['average']['actual']) == 156
    assert int(ci.summary_data['average']['predicted']) == 129
    assert int(ci.summary_data['average']['predicted_lower']) == 102
    assert int(ci.summary_data['average']['predicted_upper']) == 156
    assert int(ci.summary_data['average']['abs_effect']) == 26
    assert round(ci.summary_data['average']['abs_effect_lower'], 1) == -0.2
    assert int(ci.summary_data['average']['abs_effect_upper']) == 53
    assert round(ci.summary_data['average']['rel_effect'], 1) == 0.2
    assert round(ci.summary_data['average']['rel_effect_lower'], 1) == 0.0
    assert round(ci.summary_data['average']['rel_effect_upper'], 1) == 0.4

    assert int(ci.summary_data['cumulative']['actual']) == 4687
    assert int(ci.summary_data['cumulative']['predicted']) == 3883
    assert int(ci.summary_data['cumulative']['predicted_lower']) == 3085
    assert int(ci.summary_data['cumulative']['predicted_upper']) == 4693
    assert int(ci.summary_data['cumulative']['abs_effect']) == 803
    assert round(ci.summary_data['cumulative']['abs_effect_lower'], 1) == -6.8
    assert int(ci.summary_data['cumulative']['abs_effect_upper']) == 1601
    assert round(ci.summary_data['cumulative']['rel_effect'], 1) == 0.2
    assert round(ci.summary_data['cumulative']['rel_effect_lower'], 1) == 0.0
    assert round(ci.summary_data['cumulative']['rel_effect_upper'], 1) == 0.4

    assert round(ci.p_value, 1) == 0.0


def test_default_causal_inferences_w_date(fix_path):
    np.random.seed(1)
    data = pd.read_csv(os.path.join(fix_path, 'google_data.csv'))
    data['date'] = pd.to_datetime(data['t'])
    data.index = data['date']
    del data['t']
    del data['date']

    pre_period = ['2016-02-20 22:41:20', '2016-02-20 22:51:20']
    post_period = ['2016-02-20 22:51:30', '2016-02-20 22:56:20']

    ci = CausalImpact(data, pre_period, post_period)
    assert int(ci.summary_data['average']['actual']) == 156
    assert int(ci.summary_data['average']['predicted']) == 129
    assert int(ci.summary_data['average']['predicted_lower']) == 102
    assert int(ci.summary_data['average']['predicted_upper']) == 156
    assert int(ci.summary_data['average']['abs_effect']) == 26
    assert round(ci.summary_data['average']['abs_effect_lower'], 1) == -0.2
    assert int(ci.summary_data['average']['abs_effect_upper']) == 53
    assert round(ci.summary_data['average']['rel_effect'], 1) == 0.2
    assert round(ci.summary_data['average']['rel_effect_lower'], 1) == 0.0
    assert round(ci.summary_data['average']['rel_effect_upper'], 1) == 0.4

    assert int(ci.summary_data['cumulative']['actual']) == 4687
    assert int(ci.summary_data['cumulative']['predicted']) == 3883
    assert int(ci.summary_data['cumulative']['predicted_lower']) == 3085
    assert int(ci.summary_data['cumulative']['predicted_upper']) == 4693
    assert int(ci.summary_data['cumulative']['abs_effect']) == 803
    assert round(ci.summary_data['cumulative']['abs_effect_lower'], 1) == -6.8
    assert int(ci.summary_data['cumulative']['abs_effect_upper']) == 1601
    assert round(ci.summary_data['cumulative']['rel_effect'], 1) == 0.2
    assert round(ci.summary_data['cumulative']['rel_effect_lower'], 1) == 0.0
    assert round(ci.summary_data['cumulative']['rel_effect_upper'], 1) == 0.4

    assert round(ci.p_value, 1) == 0.0
