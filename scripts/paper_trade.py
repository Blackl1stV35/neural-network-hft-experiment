"""Paper trading / live trading main loop.

Usage:
    python scripts/paper_trade.py --config configs/deployment/production.yaml
    python scripts/paper_trade.py --mode paper --synthetic
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from loguru import logger

from src.data.preprocessing import WindowMinMaxScaler
from src.inference.onnx_engine import ONNXInferenceEngine
from src.risk.circuit_breaker import CircuitBreaker, PositionSizer
from src.risk.uncertainty import UncertaintyMonitor
from src.execution.order_manager import OrderManager
from src.monitoring.alerts import MetricsCollector, TelegramAlerter
from src.utils.config import load_env
from src.utils.logger import setup_logger


class TradingLoop:
    """Main trading loop that ties everything together."""

    def __init__(self, config_path: str, synthetic: bool = False):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.synthetic = synthetic
        self.running = False
        self.hitl_enabled = False

        # Components
        self.engine: ONNXInferenceEngine = None
        self.broker = None
        self.circuit_breaker: CircuitBreaker = None
        self.order_manager: OrderManager = None
        self.metrics: MetricsCollector = None
        self.alerter: TelegramAlerter = None
        self.scaler = WindowMinMaxScaler(window_size=120)

        # Data buffer
        self.price_buffer: list[float] = []
        self.feature_buffer: list[list[float]] = []
        self.buffer_size = 240  # max sequence length

    def initialize(self) -> bool:
        """Initialize all components."""
        load_env()
        risk_cfg = self.config.get("risk", {})

        # Inference engine
        model_path = self.config.get("inference", {}).get("model_path", "exports/best_model.onnx")
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}. Run export_model.py first.")
            return False

        self.engine = ONNXInferenceEngine(
            model_path,
            device=self.config.get("inference", {}).get("device", "cpu"),
            n_threads=4,
        )

        # Risk management
        self.circuit_breaker = CircuitBreaker(
            max_daily_drawdown_pct=risk_cfg.get("max_daily_drawdown_pct", 2.0),
            max_consecutive_losses=risk_cfg.get("max_consecutive_losses", 5),
            latency_kill_ms=risk_cfg.get("latency_kill_ms", 50.0),
        )

        sizer = PositionSizer(max_risk_pct=risk_cfg.get("max_position_risk_pct", 2.0))
        uncertainty = UncertaintyMonitor()

        # Broker
        if self.synthetic:
            self.broker = SyntheticBroker()
        else:
            from src.execution.broker_mt5 import MT5Broker
            from src.utils.config import BrokerConfig

            bc = BrokerConfig.from_env()
            self.broker = MT5Broker()
            if not self.broker.connect(bc.login, bc.password, bc.server, bc.path):
                logger.error("Failed to connect to broker")
                return False

        # Order manager
        self.order_manager = OrderManager(
            broker=self.broker,
            circuit_breaker=self.circuit_breaker,
            position_sizer=sizer,
            uncertainty_monitor=uncertainty,
            base_lot_size=self.config.get("broker", {}).get("lot_size", 0.01),
        )

        # HITL exit approval
        self.hitl_enabled = self.config.get("risk", {}).get("human_exit_approval", False)
        if self.hitl_enabled:
            logger.info("HITL exit approval ENABLED — exits will require human confirmation")

        # Monitoring
        self.metrics = MetricsCollector()
        self.alerter = TelegramAlerter()

        if self.config.get("monitoring", {}).get("telegram_enabled"):
            self.alerter.alert_startup()

        logger.info("All components initialized successfully")
        return True

    def run(self) -> None:
        """Main trading loop."""
        self.running = True
        mode = self.config.get("mode", "paper")
        logger.info(f"Starting trading loop (mode={mode})")

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        tick_count = 0
        while self.running:
            try:
                tick = self.broker.get_tick() if not self.synthetic else self._synthetic_tick()
                if tick is None:
                    time.sleep(0.1)
                    continue

                tick_count += 1
                self.metrics.record_tick()

                # Update buffers
                self.price_buffer.append(tick["bid"])
                self.feature_buffer.append(
                    [
                        tick["bid"],
                        tick["bid"] + 0.5,
                        tick["bid"] - 0.5,
                        tick["bid"],
                        100,
                        tick.get("spread", 20),
                    ]
                )

                # Keep buffer bounded
                if len(self.price_buffer) > self.buffer_size:
                    self.price_buffer = self.price_buffer[-self.buffer_size :]
                    self.feature_buffer = self.feature_buffer[-self.buffer_size :]

                # Need enough data for a sequence
                seq_length = 120
                if len(self.feature_buffer) < seq_length:
                    continue

                # Prepare features
                features = np.array(self.feature_buffer[-seq_length:], dtype=np.float32)
                scaled = self.scaler.transform(features).astype(np.float32)
                input_seq = scaled.reshape(1, seq_length, -1)

                # Inference
                action, confidence, latency = self.engine.predict_action(x=input_seq)

                # Process through order manager
                result = self.order_manager.process_signal(
                    action=action,
                    model_uncertainty=1.0 - confidence,
                    inference_latency_ms=latency,
                )

                # HITL: intercept automated exits for human approval
                if (
                    self.hitl_enabled
                    and result.get("action_taken") in ("uncertainty_exit", "force_close")
                    and not self.synthetic
                ):
                    pos_dir = self.order_manager.state.current_position_direction
                    ctx_dir = "LONG" if pos_dir == 1 else "SHORT" if pos_dir == -1 else "FLAT"
                    reason = result.get("block_reason", result.get("action_taken", ""))
                    print(f"\n{'=' * 50}")
                    print(f"  EXIT APPROVAL REQUESTED")
                    print(f"  Position:  {ctx_dir}")
                    print(f"  Price:     {tick['bid']:.2f}")
                    print(f"  Reason:    {reason}")
                    print(f"{'=' * 50}")
                    resp = input("  Approve exit? (y/n): ").strip().lower()
                    if resp not in ("y", "yes"):
                        logger.info("HITL: Exit vetoed by human operator")
                        result["action_taken"] = "hitl_vetoed"

                self.metrics.record_inference(latency, 1.0 - confidence)

                # Log periodically
                if tick_count % 100 == 0:
                    status = self.order_manager.get_status()
                    logger.info(
                        f"Tick {tick_count} | "
                        f"Price: {tick['bid']:.2f} | "
                        f"Action: {['SELL', 'HOLD', 'BUY'][action]} ({confidence:.2f}) | "
                        f"Latency: {latency:.1f}ms | "
                        f"Position: {status['position']}"
                    )

            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                self.metrics.connection_errors.inc() if self.metrics._prom_available else None
                time.sleep(1)

        self._shutdown()

    def _synthetic_tick(self) -> dict:
        """Generate a synthetic tick for testing."""
        if not self.price_buffer:
            price = 2000.0
        else:
            price = self.price_buffer[-1] + np.random.normal(0, 0.1)
        time.sleep(0.01)  # simulate tick rate
        return {"bid": price, "ask": price + 0.02, "spread": 20}

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info(f"Shutdown signal received ({signum})")
        self.running = False

    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        if self.alerter:
            self.alerter.alert_shutdown("normal")
        if hasattr(self.broker, "disconnect"):
            self.broker.disconnect()
        logger.info("Shutdown complete")


class SyntheticBroker:
    """Fake broker for paper trading / testing."""

    def __init__(self):
        self._positions = {}
        self._next_ticket = 1000

    def get_tick(self):
        return None  # handled by _synthetic_tick

    def buy(self, volume, comment=""):
        from src.execution.broker_mt5 import OrderResult

        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions[ticket] = {"type": "buy", "volume": volume}
        return OrderResult(success=True, ticket=ticket, price=2000.0, volume=volume, latency_ms=0.1)

    def sell(self, volume, comment=""):
        from src.execution.broker_mt5 import OrderResult

        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions[ticket] = {"type": "sell", "volume": volume}
        return OrderResult(success=True, ticket=ticket, price=2000.0, volume=volume, latency_ms=0.1)

    def close_position(self, ticket):
        from src.execution.broker_mt5 import OrderResult

        self._positions.pop(ticket, None)
        return OrderResult(success=True, ticket=ticket, price=2000.0, latency_ms=0.1)

    def get_open_positions(self):
        return [{"ticket": t, **p} for t, p in self._positions.items()]

    def get_account_info(self):
        return {"balance": 10000, "equity": 10000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deployment/production.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    setup_logger()

    loop = TradingLoop(args.config, synthetic=args.synthetic)
    if loop.initialize():
        loop.run()
    else:
        logger.error("Initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
