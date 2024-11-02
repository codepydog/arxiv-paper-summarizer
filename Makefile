test:
	poetry run pytest -vv tests \
	--cov=. \
	--cov-report=term \
	--cov-report=xml:coverage.xml

e2e-test:
	poetry run pytest -vv e2e

install:
	poetry install

format:
	poetry run isort src
	poetry run black src

lint:
	poetry run isort --check-only src
	poetry run black --check src
	poetry run flake8 src --max-line-length=150
	poetry run mypy src --ignore-missing-imports --check-untyped-defs 
