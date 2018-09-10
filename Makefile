init:
	pip install pipenv --upgrade
	pipenv install --dev --skip-lock

isort:
	pipenv run isort -rc causalimpact
	pipenv run isort -rc tests

flake8:
	pipenv run flake8 causalimpact

coverage:
	pipenv run py.test --cov-config .coveragerc --cov-report html --cov=causalimpact

.PHONY: flake8 isort coverage
