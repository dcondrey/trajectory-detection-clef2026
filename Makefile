.PHONY: setup train-s1 train-s2 predict evaluate test lint clean help

PYTHON ?= python
S1_DATA ?= data/subtask1
S2_DATA ?= data/subtask2
INPUT ?= /tmp/rtd-input
OUTPUT ?= /tmp/rtd-output

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install package and dependencies
	pip install -e ".[dev]"
	@echo "Setup complete."

train-s1: ## Train Subtask 1 source detection model
	$(PYTHON) train_s1.py --data $(S1_DATA)

train-s2: ## Train Subtask 2 safety classification model
	$(PYTHON) train_s2.py --data $(S2_DATA)

train: train-s1 train-s2 ## Train both subtask models

predict: ## Run inference on input data
	$(PYTHON) predict.py -i $(INPUT) -o $(OUTPUT)

predict-llm: ## Run LLM ensemble inference (requires API keys)
	$(PYTHON) predict.py -i $(INPUT) -o $(OUTPUT) --mode llm

evaluate: ## Evaluate predictions
	$(PYTHON) -m rtd.evaluate $(OUTPUT)

test: ## Run tests
	$(PYTHON) -c "from rtd import features, data_loader, evaluate, refusal_detector; print('All modules OK')"
	$(PYTHON) -c "from llm_ensemble import source_detection, safety_classification; print('LLM ensemble OK')"

lint: ## Run code quality checks
	$(PYTHON) -m ruff check rtd/ llm_ensemble/ *.py || true

clean: ## Remove generated files
	rm -rf __pycache__ rtd/__pycache__ llm_ensemble/__pycache__ .mypy_cache
	rm -rf cache/ models/*.txt

all: setup train predict evaluate ## Full pipeline
