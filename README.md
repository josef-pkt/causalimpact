# Causal Impact
This repository is a Python version of [Google's Causal Impact](https://github.com/google/CausalImpact) model implemented in R with all functionalities fully ported and tested. As far as results goes both should be quite similar despite the different techniques used on each one (in R package the fitting process is based on Markov Chain Monte Carlo procedure whereas in this package the model is fit by adapting a Kalman Filter as implemented in the `statsmodels` package).

## How it works
The main goal of the algorithm is to infer  the expected effect a given intervention (or any action) had on some response variable by analyzing differences between expected and observed time series data.

Data is divided in two parts: the first one is what is known as the "pre-intervention" period and the concept of [Bayesian Structural Time Series](https://en.wikipedia.org/wiki/Bayesian_structural_time_series)  is used to fit a model that best explains what has been observed. The fitted model is used in the second part of data ("post-intervention" period) to forecast what the response would look like had the intervention not taken place. The inferences are based on the differences between observed response to the predicted one which yields the absolute and relative expected effect the intervention caused on data.

The model makes as assumption (which is recommended to be confirmed in your data) that the response variable can be precisely modeled by a linear regression with what is known as "covariates" (or `X`) that **must not** be affected by the intervention that took place (for instance, if a company wants to infer what impact a given marketing campaign will have on its "revenue", then its daily "visits" cannot be used as a covariate as probably the total visits might be affected by the campaign. 

The model is more commonly used to infer the impact that marketing interventions have on businesses such as the expected revenue associated to a given campaign or even to assert more precisely the revenue a given channel brings in by completely turning it off (also known as "hold-out" tests). It's important to note though that the model can be extensively used in different areas and subjects; any intervention on time series data can potentially be modeled and inferences be made upon observed and predicted data.

Please refer to *getting started* in the `examples` folder for more information.

## Instalation
    pip install pycausalimpact
or (recommended):

    pipenv install pycausalimpact
## Requirements

 - Python 3.6
 - statsmodels 0.9.0
 - matplotlib
 - jinja2

## Getting Started
We recommend this [presentation](https://www.youtube.com/watch?v=GTgZfCltMm8) by Kay Brodersen (one of the creators of the causal impact implementation in R).

We also created this introductory ipython notebook with examples of how to use the package.

## Contributing, Bugs, Questions
Contributions are more than welcome! If you want to propose new changes, fix bugs or improve something feel free to fork the repository and send us a Pull Request. You can also open new `Issues` for reporting bugs and general problems.
