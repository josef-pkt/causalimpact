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
General fixtures for tests.
"""
import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fix_path():
    p = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(p, 'fixtures')
    return p


@pytest.fixture
def rand_data():
    return pd.DataFrame(np.random.randn(200, 3), columns=["y", "x1", "x2"])


@pytest.fixture
def date_rand_data(rand_data):
    date_rand_data = rand_data.set_index(pd.date_range(
        start='20180101',
        periods=len(rand_data))
    )
    return date_rand_data


@pytest.fixture
def pre_int_period():
    return [0, 99]


@pytest.fixture
def post_int_period():
    return [100, 199]


@pytest.fixture
def pre_str_period():
    return ['20180101', '20180410']


@pytest.fixture
def post_str_period():
    return ['20180411', '20180719']
