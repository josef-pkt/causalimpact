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

import pandas as pd

from causalimpact.misc import unstandardize


class Inferences(object):
    """All computations related to the inference process of the post-intervention
    prediction is handled through the methods implemented here.

    Args
    ----
      inferences: pandas DataFrame with all the necessary information for running the
          final analysis for causal impact. The columnas are:
              'cum_post_y': culmulative response ``y``.
              'preds': predictions for pre and post data.
              'preds_lower': lower boundary of predictions.
              'preds_upper': upper boundary for predictions.
              'cum_post_pred': cumulative of predictions in post data.
              'cum_post_pred_lower': cumulative of lower boundary predictions in post
                  data.
              'cum_post_pred_upper': cumulative of upper boundary predictions in post
                  data.
              'point_effects': the difference between predicted data and observed ``y``.
              'point_effects_lower': difference between lower predicted data and
                  observed ``y``.
              'point_effects_upper': difference between upper predicted data and
                  observed ``y``.
              'cum_effects': cumulative of point effects in post data.
              'cum_effects_lower': cumulative of lower point effects in post data.
              'cum_effects_upper': cumulative of upper point effects in post data.
    """
    def __init__(self, inferences=None):
        self.inferences = None

    def _unstardardize(self, data):
        """If input data was standardized, this method is used to bring back data to its
        original form. The parameter `self.mu_sig` from `main.BaseCausal` holds the values
        used for normalization (average and std, respectively). In case `self.mu_sig` is
        None, it means no standardization was applied; in this case we just return data.

        Args
        ----
          self:
            mu_sig: Tuple where first value is the mean and second is the standard
                deviation used for normalization.
          data: input vector to apply unstardization

        Returns
        -------
          ``data`` if `self.mu_sig` is None; returns the unstandizated data otherwise.
        """
        if self.mu_sig is None:
            return data
        return unstandardize(data, self.mu_sig) 
        
    def compile_posterior_inferences(self):
        """Runs the posterior causal impact inference computation using the already
        trained model.

        Args
        ----
          self:
            trained_model: ``UnobservedComponentsResultsWrapper``.
            pre_data: pandas DataFrame.
            post_data: pandas DataFrame.
            alpha: float.
            mu_sig: tuple where first value is the mean used for standardization and second
                value is the standard deviation.
        """
        pre_predictor = self.trained_model.get_prediction()
        post_predictor =self.trained_model.get_forecast(
            steps=len(self.post_data),
            exog=post_data.iloc[:, 1].values,
            alpha=self.alpha
        )
        pre_preds = self._unstardardize(pre_predictor.predicted_mean, self.mu_sig)
        post_preds = self._unstardardize(post_predictor.predicted_mean, self.mu_sig)
        # Sets index properly
        pre_preds.index = self.pre_data.index
        post_preds.index = self.post_data.index
        # Confidence Intervals
        pre_ci = self._unstardardize(pre_predictor.conf_int(alpha=self.alpha)
        pre_preds_lower = pre_ci.iloc[:, 0] # Only valid from statsmodels 0.9.0
        pre_preds_upper = pre_ci.iloc[:, 1]
        post_ci = self._unstardardize(post_predictor.conf_int(alpha=self.alpha)
        post_preds_lower = post_ci[:, 0]
        post_preds_upper = post_ci[:, 1]
        # Sets index properly
        pre_preds_lower.index = self.pre_data.index
        pre_preds_upper.index = self.pre_data.index
        post_preds_lower.index = self.post_data.index
        post_preds_upper.index = self.post_data.index
       # Concatenations
        preds = pd.concat([pre_preds, post_preds])
        preds_lower = pd.concat([pre_preds_lower, post_preds_lower])
        preds_upper = pd.concat([pre_preds_upper, post_preds_upper])
        # Cumulative analysis
        post_cum_y = np.cumsum(self.post_data.iloc[:, 0])
        post_cum_pred = np.cumsum(post_preds)
        post_cum_pred_lower = np.cumsum(post_preds_lower)
        post_cum_pred_upper = np.cumsum(post_preds_upper)
        # Effects analysis
        point_effects = self.data.iloc[:, 0] - preds
        point_effects_lower = self.data.iloc[:, 0] - preds_lower
        point_effects_upper = self.data.iloc[:, 0] - preds_upper
        # Cumulative Effects analysis
        cum_effects_lower = np.cumsum(post_preds_lower)
        cum_effects_upper = np.cumsum(post_preds_upper)
        self.inferences = pd.concat(
            [
                cum_post_y,
                preds,
                post_preds,
                preds_lower,
                preds_upper,
                post_cum_pred,
                post_cum_pred_lower,
                post_cum_pred_upper,
                point_effects,
                point_effects_lower,
                point_effects_upper,
                cum_effects,
                cum_effects_lower,
                cum_effects_upper
            ]
        )
        self.inferences.index = [
            'cum_post_y',
            'preds',
            'preds_lower',
            'preds_upper',
            'cum_post_pred',
            'cum_post_pred_lower',
            'cum_post_pred_upper',
            'point_effects',
            'point_effects_lower',
            'point_effects_upper',
            'cum_effects',
            'cum_effects_lower',
            'cum_effects_upper'
        ]

    def summarize_posterior_inference(self):
        """After running the posterior inferences compilation, this method aggregates
        the results and gets the final interpretation for the causal impact results, such
        as what was the observed absolute impact of the given intervention.

        Raises
        ------
          RuntimeError: if ``self.inferences`` is None, meaning the inferences compilation
                        was not processed yet.
        """
        infers = self.inferences
        if infers is None:
            raise RuntimeError('First run inferences compilation.')
        mean_post_y = self.post_data.iloc[:, 0].mean()
        mean_pred = infers['
        


