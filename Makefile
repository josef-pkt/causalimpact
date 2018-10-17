init:
	pip install pipenv --upgrade
	pipenv install --dev --skip-lock

isort:
	pipenv run isort -rc causalimpact
	pipenv run isort -rc tests

isort-check:
	pipenv run isort -rc -c causalimpact
	pipenv run isort -rc -c tests

flake8:
	pipenv run flake8 causalimpact

coverage:
	pipenv run py.test -p no:warnings --cov-config .coveragerc --cov-report html --cov-report term --cov-report xml --cov=causalimpact

test:
	pipenv run py.test

.PHONY: flake8 isort coverage test
