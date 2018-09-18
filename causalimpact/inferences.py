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
Computes posterior inferences related to post-intervention period of a time series
based model.
"""

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.structural import UnobservedComponents

from causalimpact.misc import unstandardize


class Inferences(object):
    """
    All computations related to the inference process of the post-intervention
    prediction is handled through the methods implemented here.
    """
    def __init__(self, n_sims=1000):
        self._inferences = None
        self._p_value = None
        self.n_sims = n_sims

    @property
    def inferences(self):
        """
        Returns pandas DataFrame of inferred inferences for post-intervention analysis.
        """
        return self._inferences

    @inferences.setter
    def inferences(self, value):
        """
        Makes attribute `inferences` Read-Only for the client.

        Args
        ----
          value: pandas DataFrame.
              General information of the inferences analysis performed in the
              post-intervention period.

        Raises
        ------
          AttributeError: if trying to set a new value to `inferences` had it already
              received the posterior analysis computation.
        """
        if self._inferences is None:
            if not isinstance(value, pd.DataFrame):
                raise ValueError('inferences must be of type pandas DataFrame')
            self._inferences = value
        else:
            raise AttributeError('inferences property is Read-Only')

    @property
    def p_value(self):
        """
        Returns the computed `p-value` for the inference analysis performed in the
        post-intervention period.
        """
        return self._p_value

    @p_value.setter
    def p_value(self, value):
        """
        Sets value for `_p-value` just once and makes sure the value is Ready-Only.

        Args
        ----
          value: float.
              Ranges between 0 and 1.

        Raises
        ------
          AttributeError: if trying to set a new value to `p_value` had it already
              received the posterior analysis computation.
        """
        if self._p_value is None:
            if value < 0 or value > 1:
                raise ValueError('p-value must range between 0 and 1')
            self._p_value = value
        else:
            raise AttributeError('p_value attribute is Read-Only.')

    def _unstardardize(self, data):
        """
        If input data was standardized, this method is used to bring back data to its
        original form. The parameter `self.mu_sig` from `main.BaseCausal` holds the values
        used for normalization (average and std, respectively). In case `self.mu_sig` is
        None, it means no standardization was applied; in this case we just return data.

        Args
        ----
          self:
            mu_sig: tuple
                First value is the mean and second is the standard deviation used for
                normalization.
          data: numpy.array
              Input vector to apply unstardization.

        Returns
        -------
          numpy.array: `data` if `self.mu_sig` is None; the unstandizated data otherwise.
        """
        if self.mu_sig is None:
            return data
        return unstandardize(data, self.mu_sig) 
        
    def compile_posterior_inferences(self):
        """
        Runs the posterior causal impact inference computation using the already
        trained model.

        Args
        ----
          self:
            trained_model: `UnobservedComponentsResultsWrapper`.
            pre_data: pandas DataFrame.
            post_data: pandas DataFrame.
            alpha: float.
            mu_sig: tuple.
                First value is the mean used for standardization and second value is the
                standard deviation.
        """
        pre_predictor = self.trained_model.get_prediction()
        post_predictor = self.trained_model.get_forecast(
            steps=len(self.post_data),
            exog=self.post_data.iloc[:, 1:].values,
            alpha=self.alpha
        )
        pre_preds = self._unstardardize(pre_predictor.predicted_mean)
        post_preds = self._unstardardize(post_predictor.predicted_mean)

        # Sets index properly
        pre_preds.index = self.pre_data.index
        post_preds.index = self.post_data.index

        # Confidence Intervals
        pre_ci = self._unstardardize(pre_predictor.conf_int(alpha=self.alpha))
        pre_preds_lower = pre_ci.iloc[:, 0] # Only valid from statsmodels 0.9.0
        pre_preds_upper = pre_ci.iloc[:, 1]
        post_ci = self._unstardardize(post_predictor.conf_int(alpha=self.alpha))
        post_preds_lower = post_ci.iloc[:, 0]
        post_preds_upper = post_ci.iloc[:, 1]

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
        post_point_effects = self.post_data.iloc[:, 0] - preds
        post_point_effects_lower = self.post_data.iloc[:, 0] - preds_lower
        post_point_effects_upper = self.post_data.iloc[:, 0] - preds_upper

        # Cumulative Effects analysis
        cum_effects = np.cumsum(post_point_effects)
        cum_effects_lower = np.cumsum(post_point_effects_lower)
        cum_effects_upper = np.cumsum(post_point_effects_upper)
        self.inferences = pd.concat(
            [
                post_cum_y,
                preds,
                post_preds,
                post_preds_lower,
                post_preds_upper,
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
            ],
            axis=1
        )

        self.inferences.columns = [
            'post_cum_y',
            'preds',
            'post_preds',
            'post_preds_lower',
            'post_preds_upper',
            'preds_lower',
            'preds_upper',
            'post_cum_pred',
            'post_cum_pred_lower',
            'post_cum_pred_upper',
            'point_effects',
            'point_effects_lower',
            'point_effects_upper',
            'cum_effects',
            'cum_effects_lower',
            'cum_effects_upper'
        ]

    def summarize_posterior_inferences(self):
        """
        After running the posterior inferences compilation, this method aggregates
        the results and gets the final interpretation for the causal impact results, such
        as what is the expected absolute impact of the given intervention.

        Raises
        ------
          RuntimeError: if `self.inferences` is `None`, meaning the inferences
              compilation was not processed yet.
        """
        infers = self.inferences
        if infers is None:
            raise RuntimeError('First run inferences compilation.')

        # Compute the mean of metrics.
        mean_post_y = self.post_data.iloc[:, 0].mean()
        mean_post_pred = infers['post_preds'].mean()
        mean_post_pred_lower = infers['post_preds_lower'].mean()
        mean_post_pred_upper = infers['post_preds_upper'].mean()

        # Compute the sum of metrics.
        sum_post_y = self.post_data.iloc[:, 0].sum()
        sum_post_pred = infers['post_preds'].sum()
        sum_post_pred_lower = infers['post_preds_lower'].sum()
        sum_post_pred_upper = infers['post_preds_upper'].sum()

        # Causal Impact analysis metrics.
        abs_effect = mean_post_pred - mean_post_y
        abs_effect_lower = mean_post_pred_lower - mean_post_y
        abs_effect_upper = mean_post_pred_upper - mean_post_y

        sum_abs_effect = sum_post_pred - sum_post_y
        sum_abs_effect_lower = sum_post_pred_lower - sum_post_y
        sum_abs_effect_upper = sum_post_pred_upper - sum_post_y

        rel_effect = abs_effect / mean_post_y
        rel_effect_lower = abs_effect_lower / mean_post_y
        rel_effect_upper = abs_effect_upper / mean_post_y

        sum_rel_effect = sum_abs_effect / sum_post_y
        sum_rel_effect_lower = sum_abs_effect_lower / sum_post_y
        sum_rel_effect_upper = sum_abs_effect_upper / sum_post_y

        # Prepares all this data into a DataFrame for later retrieval, such as when
        # running the `summary` method.
        summary_data = [
            [mean_post_y, sum_post_y],
            [mean_post_pred, sum_post_pred],
            [mean_post_pred_lower, sum_post_pred_lower],
            [mean_post_pred_upper, sum_post_pred_upper],
            [abs_effect, sum_abs_effect],
            [abs_effect_lower, sum_abs_effect_lower],
            [abs_effect_upper, sum_abs_effect_upper],
            [rel_effect, sum_rel_effect],
            [rel_effect_lower, sum_rel_effect_lower],
            [rel_effect_upper, sum_rel_effect_upper]
        ]
        self.summary_data = pd.DataFrame(
            summary_data,
            columns=['average', 'cumulative'],
            index=[
                'actual',
                'predicted',
                'predicted_lower',
                'predicted_uppper', 
                'abs_effect',
                'abs_effect_lower',
                'abs_effect_upper',
                'rel_effect',
                'rel_effect_lower',
                'rel_effect_upper'
            ]
        )
        # We also save the p-value which will be used in `summary` as well.
        self.p_value = self.compute_p_value()

    def compute_p_value(self, n_sims=1000):
        """
        Computes the p-value for the hypothesis testing that there's signal in the
        observed data. The computation follows the same idea as the one implemented in R
        by Google which consists of simulating with the fitted parameters several time
        series for the post-intervention period and counting how many either surpass the
        total summation of `y` (in case there's positive relative effect) or how many
        falls under its summation (in which case there's negative relative effect).

        For a better understanding of how this solution was obtained, this discussion was 
        used as the main guide:

        https://stackoverflow.com/questions/51881148/simulating-time-series-with-unobserved-components-model/ # noqa

        Args
        ----
          n_sims: int.
              Representing how many simulations to run for computing the p-value.

        Returns
        -------
          p_value: float.
              Ranging between 0 and 1, represents the likelihood of obtaining the observed
              data by random chance.
        """
        # For more information about the `trend` and how it works, please refer to:
        # https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html #noaq
        trend = self.model.trend_specification
        y = np.zeros(len(self.post_data))
        X = self.post_data.iloc[:, 1:] if self.post_data.shape[1] > 1 else None
        model = UnobservedComponents(y, level=trend, exog=X)
        # `params` is related to the parameters found when fitting the Kalman filter
        # from the observed time series.
        params = self.trained_model.params
        predicted_state = self.trained_model.predicted_state[..., -1]
        predicted_state_cov = self.trained_model.predicted_state_cov[..., -1]
        y_post_sum = self.post_data.iloc[:, 0].sum()
        positive_signal, negative_signal = 1, 1
        for _ in range(n_sims):
            initial_state = np.random.multivariate_normal(predicted_state,
                                                          predicted_state_cov)
            sim = model.simulate(params, len(self.post_data), initial_state=initial_state)
            sim_sum = sim.sum()
            if sim_sum > y_post_sum:
                positive_signal += 1
            else:
                negative_signal += 1
        # The minimum value between positive and negative signals reveals how many times
        # either the summation of the simulation could surpass ``y_post_sum`` or be
        # surpassed by the same (in which case it means the sum of the simulated time
        # series is bigger than ``y_post_sum`` most of the time, meaning the signal in
        # this case reveals the impact caused the response variable to decrease from what
        # was expected had no effect taken place.
        p_value = min(positive_signal, negative_signal) / (n_sims + 1)
        return p_value
