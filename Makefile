.PHONY: setup test lint data train backtest export paper-trade clean docker

# Setup
setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -e ".[dev]"
	cp -n .env.example .env || true
	@echo "Setup complete. Activate with: source .venv/bin/activate"

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Linting
lint:
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

# Data
data-synthetic:
	python scripts/download_data.py --synthetic --days 90

data-mt5:
	python scripts/download_data.py --symbol XAUUSD --timeframe M1 --days 365

# FT.com sentiment data
scrape-ft:
	python scripts/scrape_ft.py --start 2024-01 --min-relevance 0.01

scrape-ft-full:
	python scripts/scrape_ft.py --start 2020-01 --min-relevance 0.005

build-embeddings:
	python scripts/build_embeddings.py --data-dir data --cache-dir data/ft_cache/processed

# Training
train-baseline:
	python scripts/train.py model=cnn_lstm data=xauusd

train-mamba:
	python scripts/train.py model=mamba_ssm data=xauusd

train-rl-sac:
	python scripts/train_rl.py --agent sac --steps 500000

train-rl-dqn:
	python scripts/train_rl.py --agent dqn --steps 300000

# Backtesting
backtest:
	python scripts/backtest.py model=cnn_lstm data=xauusd

# Export
export:
	python scripts/export_model.py --model cnn_lstm --checkpoint models/cnn_lstm_best.pt --benchmark

# Paper trading
paper-trade:
	python scripts/paper_trade.py --synthetic

paper-trade-mt5:
	python scripts/paper_trade.py --config configs/deployment/production.yaml

# Docker
docker-build:
	docker build -t xauusd-trading:latest .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data xauusd-trading:latest

# Rust inference
rust-build:
	cd rust_inference && cargo build --release

rust-test:
	cd rust_inference && cargo test

# Clean
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf outputs/ multirun/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf .venv data/ models/ exports/ logs/ wandb/
