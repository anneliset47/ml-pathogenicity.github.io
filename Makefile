PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: setup setup-full reproduce reproduce-full clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && python -m pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt

setup-full: setup
	$(ACTIVATE) && pip install -r requirements-optional.txt

reproduce:
	$(ACTIVATE) && python run_pipeline.py --all

reproduce-full:
	$(ACTIVATE) && python run_pipeline.py --all --with-nn

clean:
	rm -rf $(VENV)
