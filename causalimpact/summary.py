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
Summarizes performance information inferred in post-inferences compilation process.
"""


import os

from jinja2 import Template

from causalimpact.misc import get_z_score

_here = os.path.dirname(os.path.abspath(__file__))
summary_tmpl_path = os.path.join(_here, 'templates', 'summary')
report_tmpl_path = os.path.join(_here, 'templates', 'report')

SUMMARY_TMPL = Template(open(summary_tmpl_path).read())
REPORT_TMPL = Template(open(report_tmpl_path).read())


class Summary(object):
    """
    Prepares final summary with causal impact results telling whether an effect has been
    identified in data or not.
    """
    def __init__(self):
        self.summary_data = None

    def summary(self, output='summary'):
        """
        Returns final results from causal impact analysis, such as absolute observed
        effect, the relative effect between prediction and observed variable, cumulative
        performances in post-intervention period among other metrics.

        Args
        ----
          output: str.
              Can be either "summary" or "report". The first is a simpler output just
              informing general metrics such as expected absolute or relative effect.

        Returns
        -------
          summary: str.
              Contains results of the causal impact analysis.

        Raises
        ------
          RuntimeError: if `self.summary_data` is None meaning the post inference
              compilation was not performed yet.
        """
        if self.summary_data is None:
            raise RuntimeError('Posterior inferences must be first computed before '
                               'running summary.')
        if output not in {'summary', 'report'}:
            raise ValueError('Please choose either summary or report for output.')
        if output == 'summary':
            summary = SUMMARY_TMPL.render(
                summary=self.summary_data.to_dict(),
                alpha=self.alpha,
                z_score=get_z_score(1 - self.alpha / 2.),
                p_value=self.p_value
            )
        else:
            summary = REPORT_TMPL.render(
                summary=self.summary_data.to_dict(),
                alpha=self.alpha,
                p_value=self.p_value
            )
        return summary
