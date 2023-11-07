.PHONY: install clean build format test

## Install for production
install:
	@echo ">> Installing dependencies"
	python -m pip install -U pip wheel
	python -m pip install -e .

## Install for development 
install-dev: install
	python -m pip install -e ".[dev]"

## Install for development 
install-api: install
	python -m pip install -e ".[api]"

## Install for development 
install-graphgym: install
	python -m pip install -e ".[graphgym]"

install-docs:
	python -m pip install -e ".[docs]"
## Build sdist and wheel
build:
	hatch build

## Delete all temporary files
clean:
	rm -rf .ipynb_checkpoints
	rm -rf **/.ipynb_checkpoints
	rm -rf .pytest_cache
	rm -rf **/.pytest_cache
	rm -rf __pycache__
	rm -rf **/__pycache__
	rm -rf build
	rm -rf dist

## Lint using ruff
ruff:
	ruff .

## Format files using black
format:
	ruff . --fix
	black .

## Run tests
test:
	pytest --cov=src --cov-report html --log-level=WARNING --disable-pytest-warnings

## Run checks (ruff + test)
check:
	ruff check .
	black --check .

## Run api
api: install-api
	python -m uvicorn openimpact.main:app --reload


