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

"""Causal Impact class for running impact inferences caused in a time evolving system."""

import numpy as np
import pandas as pd


class CausalImpact:
    """Main class used to run the Causal Impact algorithm implemented by Google as
    described in their paper:

    https://google.github.io/CausalImpact/CausalImpact.html

    The main difference between Google's R package and Python's is that in the latter the
    optimization will be performed by using Kalman Filters as implemented in `statsmodels`
    package, contrary to the Markov Chain Monte Carlo technique used in R.

    Despite the different techniques, results should converge to the same optimum state
    space.

    Args
    ----
      data: Can be either a numpy array or a pandas DataFrame where the first column must
            contain the `y` measured value while the others contain the covariates
            `X` that are used in the linear regression component of the model.

      pre_period: A list of size two containing either `int` or `str` values that
                  references the first time point in the trained data up to the last one
                  that will be used in the pre-intervention period for training the
                  model. For example, valid inputs are: `[0, 30]` or
                  `['20180901', '20180930']`. The latter can be used only if the input
                  `data` is a pandas DataFrame whose index is time based.
                  Ideally, it should slice the data up to when the intervention happened
                  so that the trained model can more precisely predict what should have
                  happened in the post-intervention period had no interference taken
                  place.

      post_period: The same as `pre_period` but references where the post-intervention
                   data begins and ends. This is the part of `data` used to make
                   inferences.

      model: An `UnobservedComponentsModel` from `statsmodels` package whose default value
             is ``None``. If a customized model is desired than this argument can be used
             otherwise a default 'local level' model is implemented. When using a user-
             defined model, it's still required to send `data` as input even though the
             pre-intervention period is already present in the model `endog` and `exog`
             attributes.

      alpha: A float that ranges between 0. and 1. indicating the significance level that
             will be used when statistically testing for signal presencen in the post-
             intervention period.

    Returns
    -------
      A CausalImpact object.

    Examples:
    ---------
      >>> import numpy as np
      >>> from statsmodels.tsa.statespace.structural import UnobservedComponents
      >>> from statsmodels.tsa.arima_process import ArmaProcess

      >>> np.random.seed(12345)
      >>> ar = np.r_[1, 0.9]
      >>> ma = np.array([1])
      >>> arma_process = ArmaProcess(ar, ma)
      >>> X = 100 + arma_process.generate_sample(nsample=100)
      >>> y = 1.2 * X + np.random.normal(size=100)
      >>> data = np.concatenate((y.reshape(-1, 1), X.reshape(-1, 1)), axis=1)
      >>> pre_period = [0, 70]
      >>> post_period = [70, 100]

      >>> ci = CausalImpact(data, pre_period, post_period)
      >>> ci.summary()
      >>> ci.summary('report')
      >>> ci.plot()

      Using pandas DataFrames:

      >>> df = pd.DataFrame(data)
      >>> df = df.set_index(pd.date_range(start='20180101', periods=len(data)))
      >>> pre_period = ['20180101', '20180311']
      >>> post_period = ['20180312', '20180410']
      >>> ci = CausalImpact(df, pre_period, post_period)

      Using a customized model:

      >>> pre_y = data[:70, 0]
      >>> pre_X = data[:70, 1:]
      >>> ucm = UnobservedComponentsModel(endog=pre_y, level='llevel', exog=pre_X)
      >>> ci = CausalImpact(data, pre_period, post_period, model=ucm)
    """
    def __init__(self, data, pre_period, post_period, model=None, alpha=0.05):
        pass

    def check_ci_input_data(self, data, pre_period, post_period, model, alpha):
        """Checks and formats when appropriate the input data for running the Causal
        Impact algorithm. Performs assertions such as missing or invalid arguments.

        Args
        ----
          data: `numpy.array` or `pandas.DataFrame`.
          pre_period: a list of size two containing either `int` or `str` values.
          post_period: the same as ``pre_period``.
          model: Either None or an UnobservedComponentsModel object.
          alpha: float.

        Returns
        -------
          data: a pandas DataFrame with validated data.
          pre_period: a list with two validated values of either `int` or `str`.
          post_period: the same as ``pre_period``.
          model: Either ``None`` or `UnobservedComponentsModel` validated to be correct.
          alpha: float ranging from 0 to 1.

        Raises
        ------
        """
        input_args = locals().copy()
        model = input_args.pop('model')
        none_args = [arg for arg, value in input_args.items() if value is None]
        if none_args:
            raise ValueError('{args} cannot be empty'.format(args=', '.join(none_args)))
        checked_data = self._format_input_data(data)

    def _format_input_data(self, data):
        """Validates and formats input data.

        Args
        ----
          data: `numpy.array` or `pandas.DataFrame`.

        Returns
        -------
          data: validated data to be used in Causal Impact algorithm.

        Raises
        ------
          ValueError: if input ``data`` is non-convertible to pandas DataFrame.
                      if input ``data`` has non-numeric values.
                      if input ``data`` has less than 3 points.
                      if input covariates have NAN values.
        """
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except ValueError as err:
                raise ValueError(
                    'Input ``data`` is not valid. Cause of error: {err}'.format(
                        err=str(err))
                )
        # Must contain only numeric values
        if not data.applymap(np.isreal).values.all():
            raise ValueError('Input ``data`` must contain only numeric values')
        # Must have at least 3 points of observed data
        if data.shape[0] < 3:
            raise ValueError('Input data must have more than 3 points')
        # Covariates cannot have NAN values
        if data.shape[1] > 1:
            if data.isna().values.all():
                raise ValueError('Input data cannot have NAN values')
        return data

    def _format_pre_post_data(self, data, pre_period, post_period):
        """Checks ``pre_period`` and ``post_period`` and returns data sliced accordingly
        to each period.

        Args
        ----
          data: pandas DataFrame.
          pre_period: list with `int` or `str` values.
          post_period: same as ``pre_period``.

        Returns
        -------
          periods: list where first value is pre-intervention data and second value is
                   post-intervention.
        Raises
        ------
        """
        pass
        # if not isinstance(

    def _check_periods(self, period, data_index):
        """Validates pre or post period inputs.

        Args
        ----
          period: a list containing two values that can be either `int` or `str`.
          data_index: index of input data such as `RangeIndex` from pandas.

        Returns
        -------
          period: validated period list.

        Raises
        ------
          ValueError: if input ``period`` is not of type `list`.
                      if input doesn't have two elements.
        """
        if not isinstance(period, list):
            raise ValueError('Input ``period`` must be of type `list`.')
        if len(period) != 2:
            raise ValueError(
                '``period`` must have two values regarding the beginning and end of '
                'the pre and post intervention data'
            )
        null_args = [d for d in period if d]
        if null_args:
            raise ValueError('Input period cannot have Null values')
        # if period contains strings, try to convert to datetime. ``data_index`` should
        # also be of DatetimeIndex type
        # if isinstance(
