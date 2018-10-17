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
Tests for summary.py module.
"""


import os

import pandas as pd
import pytest

from causalimpact.summary import Summary


@pytest.fixture
def summary_data():
    data = [
        [5.343, 10.343],
        [4.343, 9.343],
        [3.343, 8.343],
        [6.343, 9.343],
        [3.343, 10.343],
        [2.343, 4.343],
        [6.343, 9.343],
        [0.123, 0.233],
        [0.143, 0.133],
        [0.343, 0.333]
    ]
    data = pd.DataFrame(
        data,
        columns=['average', 'cumulative'],
        index=[
            'actual',
            'predicted',
            'predicted_lower',
            'predicted_upper', 
            'abs_effect',
            'abs_effect_lower',
            'abs_effect_upper',
            'rel_effect',
            'rel_effect_lower',
            'rel_effect_upper'
        ]
    )
    return data


@pytest.fixture
def summarizer():
    return Summary()


def test_summary_raises(summarizer):
    summarizer = Summary()
    with pytest.raises(RuntimeError):
        summarizer.summary()
 
    with pytest.raises(ValueError):
        summarizer.summary_data = 'test'
        summarizer.summary('test')    


def test_output_summary_1(summary_data, fix_path, summarizer):
    summarizer.summary_data = summary_data
    summarizer.alpha = 0.1
    summarizer.p_value = 0.459329

    result = summarizer.summary()
    expected = open(os.path.join(fix_path, 'test_summary_output_1')).read().strip()
    assert result == expected


def test_summary_1(summary_data, fix_path, summarizer):
    # detected positive signal but with no significance.
    summarizer.summary_data = summary_data
    summarizer.alpha = 0.1
    summarizer.p_value = 0.5
    summary_data['average']['rel_effect'] = 0.41
    summary_data['average']['rel_effect_lower'] = -0.30
    summary_data['average']['rel_effect_upper'] = 0.30
 
    result = summarizer.summary(output='report')
    expected = open(os.path.join(fix_path, 'test_summary_1')).read().strip()
    assert result == expected


def test_summary_2(summary_data, fix_path, summarizer):
    # detected positive signal with significance.
    summarizer.summary_data = summary_data
    summarizer.alpha = 0.1
    summarizer.p_value = 0.05
    summary_data['average']['rel_effect'] = 0.41
    summary_data['average']['rel_effect_lower'] = 0.434
    summary_data['average']['rel_effect_upper'] = 0.234
 
    result = summarizer.summary(output='report')
    expected = open(os.path.join(fix_path, 'test_summary_2')).read().strip()
    assert result == expected


def test_summary_3(summary_data, fix_path, summarizer):
    # detected negative signal but with no significance.
    summary_data['average']['rel_effect'] = -0.343
    summary_data['average']['rel_effect_lower'] = -0.434
    summary_data['average']['rel_effect_upper'] = 0.234
    summarizer.summary_data = summary_data
    summarizer.alpha = 0.1
    summarizer.p_value = 0.5

    result = summarizer.summary(output='report')
    expected = open(os.path.join(fix_path, 'test_summary_3')).read().strip()
    assert result == expected


def test_summary_4(summary_data, fix_path, summarizer):
    # detected negative signal with significance.
    summary_data['average']['rel_effect'] = -0.343
    summary_data['average']['rel_effect_lower'] = -0.434
    summary_data['average']['rel_effect_upper'] = -0.234
    summarizer.summary_data = summary_data
    summarizer.alpha = 0.1
    summarizer.p_value = 0.05

    result = summarizer.summary(output='report')
    expected = open(os.path.join(fix_path, 'test_summary_4')).read().strip()
    assert result == expected
