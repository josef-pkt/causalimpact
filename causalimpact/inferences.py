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

"""Computes posterior inferences related to post-intervention period of a time series
based model."""

from causalimpact.misc import unstandardize


class Inferences(object):
    """All computations related to the inference process of the post-intervention
    prediction is handled through the methods implemented here.
    """
    def compile_posterior_inferences(
            self,
            trained_model,
            pre_data,
            post_data,
            alpha,
            mu_sig
        ):
        """Runs the posterior causal impact inference computation using the already
        trained model.

        Args
        ----
          trained_model: ``UnobservedComponentsResultsWrapper``.
          pre_data: pandas DataFrame.
          post_data: pandas DataFrame.
          alpha: float.
          mu_sig: tuple where first value is the mean used for standardization and second
              value is the standard deviation.

        Returns
        -------
        """
        pre_predict = trained_model.get_prediction()
        post_predict =trained_model.get_forecast(
            steps=len(post_data),
            exog=post_data.iloc[:, 1].values,
            alpha=alpha
        )
        # If `mu_sig` is not None then data has been standardized
        if mu_sig:
            pre_predict = unstandardize(pre_predict.predicted_mean, mu_sig)
            post_predict = unstandardize(post_predict.predicted_mean, mu_sig)
            
