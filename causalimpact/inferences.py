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
    def compile_posterior_inferences(self):
        """Run the posterior inference computation."""
        pre_predictions = self.trained_model.get_prediction()
        post_predictions = self.trained_model.get_forecast(
            steps=len(self.post_data),
            exog=self.post_data.iloc[:, 1],
            alpha=self.alpha
        )
        # If `self.mu_sig` is not None then data has been standardized
        if self.mu_sig:
            pre_predictions = unstandardize(pre_predictions.predicted_mean, self.mu_sig)
            post_predictions = unstandardize(post_predictions.predicted_mean, self.mu_sig)
            
