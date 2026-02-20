PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: setup setup-dev setup-full reproduce reproduce-full lint typecheck test quality clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && python -m pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt

setup-dev: setup
	$(ACTIVATE) && pip install -r requirements-dev.txt

setup-full: setup
	$(ACTIVATE) && pip install -r requirements-optional.txt

reproduce:
	$(ACTIVATE) && python run_pipeline.py --all

reproduce-full:
	$(ACTIVATE) && python run_pipeline.py --all --with-nn

lint:
	$(ACTIVATE) && ruff check run_pipeline.py tests

typecheck:
	$(ACTIVATE) && mypy

test:
	$(ACTIVATE) && pytest

quality: lint typecheck test

clean:
	rm -rf $(VENV)
