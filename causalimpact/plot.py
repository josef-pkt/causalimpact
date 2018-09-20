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
Plots the analysis obtained in causal impact algorithm.
"""


import matplotlib.pyplot as plt
plt.figure(figsize=(15, 12))

class Plot(object):
    """Takes all the vectors and final analysis performed in the post-period inference
    to plot final graphics.
    """
    def plot(self, panels=['original', 'pointwise', 'cumulative']):
        """Plots inferences results related to causal impact analysis.

        Args
        ----
          panels: list, indicates which plot should be considered in the plot.

        Raises
        ------
          RuntimeError: if inferences were not computed yet.
        """
        if self.summary_data is None:
            raise RuntimeError('Please first run inferences before plotting results')

        inferences = self.inferences
        intervention_idx = self.post_data.index.get_loc(self.post_period[0])
        n_panels = len(panels)
        fig = plt.figure()
        ax = plt.subplot(n_panels, 1, 1)

        if 'original' in panels:
            ax.plot(self.data.iloc[:, 0], 'k', label='y')
            ax.plot(inferences['preds'], 'b--', label='Predicted')
            ax.axvline(intervention_idx, c='k', linestyle='--')
            ax.fill_between(
                inferences['preds'].index,
                inferences['preds_lower'],
                inferences['preds_uppwer'],
                facecolor='blue',
                interpolate=True,
                alpha=0.25
            )
            #ax.setp(ax.get_xticklabels(), visible=True if n_panels == 1 else False)
            ax.legend()

        if 'pointwise' in panels:
            ax = plt.subplot(n_panels, 1, 2, sharex=ax)
            ax.plot(inferences['point_effects'], 'b--', label='Point Effects')
            ax.axvline(intervention_idx, c='k', linestyle='--')
            ax.fill_between(
                inferences['point_effects'].index,
                inferences['point_effects_lower'],
                inferences['point_effects_upper'],
                facecolor='blue',
                interpolate=True,
                alpha=0.25
            )
            #ax.setp(ax.getxticklabels(), visible=True if n_panels == 2 else False)
            ax.legend()

        if 'cumulative' in panels:
            ax = plt.subplot(n_panels, 1, 3, sharex=ax)
            ax.plot(inferences['cum_effects'], 'b--', label='Cumulative Effect')
            ax.axvline(intervention_idx, c='k', linestyle='--')
            ax.fill_between(
                inferences['cum_effects'].index,
                inferences['cum_effects_lower'],
                inferences['cum_effects_upper'],
                facecolor='blue',
                interpolate=True,
                alpha=0.25
            )
            #ax.setp(ax.getxticklabels(), visible=True if n_panels == 2 else False)
            ax.legend()
        plt.show()
