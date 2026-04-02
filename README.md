# XAUUSD Neural Network Futures Trading System

A deep learning-based trading system for XAUUSD (Gold) futures combining Mamba SSM sequence modeling, hierarchical reinforcement learning, and NLP sentiment analysis.

## Architecture

```
Layer 4: Execution & Risk    → Broker API, Order Manager, Circuit Breakers
Layer 3: Decision Engine     → Meta-Policy Regime Router → Expert RL Agents
Layer 2: Feature Layer       → Mamba SSM + TCN + FinBERT + TA Indicators
Layer 1: Data Ingestion      → Tick Feed → Preprocessing → Feature Store
```

## Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install requirements
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your broker credentials and API keys
```

### 3. Download Data
```bash
python scripts/download_data.py --symbol XAUUSD --timeframe M1 --days 365
```

### 4. Train Baseline Model
```bash
python scripts/train.py model=cnn_lstm data=xauusd
```

### 5. Backtest
```bash
python scripts/backtest.py model=cnn_lstm data=xauusd
```

### 6. Paper Trade
```bash
python scripts/paper_trade.py --config configs/deployment/production.yaml
```

## Project Structure

```
xauusd-trading/
├── configs/          # Hydra configuration files
├── src/              # Core source code
│   ├── data/         # Data ingestion, preprocessing, feature engineering
│   ├── models/       # Neural network architectures
│   ├── rl/           # Reinforcement learning agents & environments
│   ├── backtesting/  # Backtesting engine with slippage modeling
│   ├── risk/         # Risk management & circuit breakers
│   ├── execution/    # Broker integration & order management
│   ├── inference/    # ONNX inference engine
│   ├── monitoring/   # Metrics, logging, Telegram alerts
│   └── utils/        # Shared utilities
├── rust_inference/   # Rust-based low-latency inference engine
├── scripts/          # Entry point scripts
├── tests/            # Unit and integration tests
└── notebooks/        # Exploration notebooks
```

## Key Design Decisions

- **CPU inference over GPU**: Single-sample ONNX inference is faster on CPU (no kernel launch overhead)
- **Polars over Pandas**: 5-10x faster tick data processing with lazy evaluation
- **Regime classifier first**: Supervised regime detection before hierarchical RL
- **Behavior cloning bootstrap**: Initialize RL agents from rule-based strategy logs
- **Shadow mode validation**: New models run alongside production before promotion

## Hardware Requirements

**Training**: GPU with 16+ GB VRAM (RTX 3090/4090, or Colab/Kaggle free GPUs)
**Inference**: Modern CPU with AVX2 support, 8+ GB RAM
**Storage**: 50+ GB for tick data (NVMe SSD recommended)

## Deployment (Free Tier)

- **Training**: Google Colab / Kaggle (free GPUs)
- **Inference**: Oracle Cloud Free Tier (4 ARM cores, 24GB RAM)
- **Tracking**: Weights & Biases free tier
- **Monitoring**: Grafana Cloud free tier
- **CI/CD**: GitHub Actions

## License

Private / Proprietary
