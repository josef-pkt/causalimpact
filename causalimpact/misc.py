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

"""Miscellaneous functions to help in the implementation of Causal Impact."""

import scipy.stats as stats


def standardize(data):
    """Applies standardization to input data. Result should have mean zero and standard
    deviation of one.

    Args
    ----
      data: pandas DataFrame.

    Returns
    -------
      list:
        data: standardized data with zero mean and std of one.
        tuple:
          mean and standard deviation used on each column of input data to make
          standardization. These values should be used to obtain the original dataframe.

    Raises
    ------
      ValueError: if data has only one value.
    """
    if data.shape[0] == 1:
        raise ValueError('Input data must have more than one value')
    mu = data.mean(skipna=True)
    std = data.std(skipna=True, ddof=0)
    data = (data - mu) / std.fillna(1)
    return [data, (mu, std)]


def unstandardize(data, mus_sigs):
    """Applies the inverse transformation to return to original data.

    Args
    ----
      data: pandas DataFrame with zero mean and std of one.
      mus_sigs: tuple where first value is the mean used for the standardization and
                second value is the respective standard deviaion.

    Returns
    -------
      data: pandas DataFrame with mean and std given by input ``mus_sigs``
    """
    mu, sig = mus_sigs
    data = (data * sig) + mu
    return data


def get_z_score(p):
    """Returns the correspondent z-score with probability area p.

    Args
    ----
      p: float ranging between 0 and 1 representing the probability area to convert.

    Returns
    -------
      The z-score correspondent of p.
    """
    return stats.norm.ppf(p)
