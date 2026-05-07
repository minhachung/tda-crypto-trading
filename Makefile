.PHONY: install data features train evaluate all test lint format clean docker docker-run

PYTHON ?= python3
DAYS ?= 90
PIP ?= pip

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PYTHON) -c "from src.binance_data import HighFreqFetcher; \
		[HighFreqFetcher(s).fetch_history(days=$(DAYS)) for s in ['BTC','ETH','SOL','ADA','DOT','LINK','AVAX']]"

features:
	$(PYTHON) examples/run_validation_v4.py $(DAYS)

train:
	$(PYTHON) examples/run_validation_v6.py $(DAYS)

evaluate:
	$(PYTHON) scripts/build_pdf.py results/RESULTS.md results/PAPER.pdf

all: data features train evaluate

walkforward:
	$(PYTHON) examples/run_validation_v8.py 365 168 0.65 0.50

rigorous:
	$(PYTHON) examples/run_validation_v9.py 365 72 30

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ examples/ tests/ scripts/

format:
	black src/ examples/ tests/ scripts/
	ruff check --fix src/ examples/ tests/ scripts/

clean:
	rm -rf data/raw/*.csv data/processed/*.npy data/persistence/*.csv
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

docker:
	docker build -t tda-crypto:latest .

docker-run:
	docker run --rm -v $(shell pwd)/results:/app/results tda-crypto:latest $(DAYS)

help:
	@echo "Targets:"
	@echo "  install      Install package + dependencies"
	@echo "  data         Fetch hourly OHLCV (use DAYS=N to override)"
	@echo "  features     Compute features (calls v4 runner)"
	@echo "  train        Run main experiment (v6 horizon sweep)"
	@echo "  evaluate     Build PDF paper from RESULTS.md"
	@echo "  walkforward  Run v8 continuous walk-forward profitability test"
	@echo "  rigorous     Run v9 holdout + ablation + permutation"
	@echo "  all          data + features + train + evaluate"
	@echo "  test         Run pytest suite"
	@echo "  lint         Run ruff"
	@echo "  format       Run black + ruff --fix"
	@echo "  clean        Remove caches and generated artifacts"
	@echo "  docker       Build Docker image"
	@echo "  docker-run   Run Docker image (mounts results/ for output)"
