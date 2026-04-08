"""Backtesting engine with realistic execution modeling.

DO NOT TRUST A MODEL THAT HASN'T BEEN BACKTESTED WITH REALISTIC:
- Spreads (XAUUSD typical: 1.5-3 pips)
- Slippage (0.5-1 pip per fill)
- Commission ($7 per round-trip lot)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from loguru import logger


@dataclass
class BacktestConfig:
    """Backtesting parameters."""

    initial_balance: float = 10_000.0
    lot_size: float = 0.01
    spread_pips: float = 2.0
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0
    pip_value: float = 0.01
    pip_usd_per_lot: float = 1.0  # per 0.01 lot
    max_position_time: int = 120
    human_exit_approval: bool = False  # HITL: require human approval for exits


@dataclass
class Trade:
    """Completed trade record."""

    entry_idx: int
    exit_idx: int
    direction: int
    entry_price: float
    exit_price: float
    pnl_pips: float
    pnl_usd: float
    hold_time: int
    entry_reason: str = ""
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Complete backtest results."""

    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_balance: float = 10_000.0
    final_balance: float = 10_000.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl_usd > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl_usd <= 0)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / max(1, self.total_trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_usd for t in self.trades)

    @property
    def avg_win(self) -> float:
        wins = [t.pnl_usd for t in self.trades if t.pnl_usd > 0]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl_usd for t in self.trades if t.pnl_usd <= 0]
        return np.mean(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_usd for t in self.trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in self.trades if t.pnl_usd <= 0))
        return gross_profit / max(gross_loss, 1e-8)

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        return float(dd.max())

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        returns = [t.pnl_usd for t in self.trades]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        # Annualize (assuming ~250 trading days, ~20 trades/day)
        return mean_ret / std_ret * np.sqrt(252 * 20)

    @property
    def sortino_ratio(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        returns = np.array([t.pnl_usd for t in self.trades])
        mean_ret = returns.mean()
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else 1e-8
        return mean_ret / downside_std * np.sqrt(252 * 20)

    def summary(self) -> str:
        return (
            f"{'=' * 50}\n"
            f"BACKTEST RESULTS\n"
            f"{'=' * 50}\n"
            f"Total Trades:    {self.total_trades}\n"
            f"Win Rate:        {self.win_rate:.1%}\n"
            f"Profit Factor:   {self.profit_factor:.2f}\n"
            f"Total PnL:       ${self.total_pnl:.2f}\n"
            f"Avg Win:         ${self.avg_win:.2f}\n"
            f"Avg Loss:        ${self.avg_loss:.2f}\n"
            f"Max Drawdown:    {self.max_drawdown:.1%}\n"
            f"Sharpe Ratio:    {self.sharpe_ratio:.2f}\n"
            f"Sortino Ratio:   {self.sortino_ratio:.2f}\n"
            f"Final Balance:   ${self.final_balance:.2f}\n"
            f"{'=' * 50}"
        )


class BacktestEngine:
    """Run backtests on historical data with realistic execution.

    Supports optional human-in-the-loop (HITL) exit approval via a callback.
    When enabled, every exit (signal reversal, max hold time) is routed through
    the callback which can veto (return False) to keep the position open.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._exit_approval_fn: Optional[Callable[[dict], bool]] = None

    def set_exit_approval_fn(self, fn: Callable[[dict], bool]) -> None:
        """Register a human-in-the-loop exit approval callback.

        The callback receives a dict with exit context and must return
        True (approve exit) or False (veto — keep position open).

        Args:
            fn: Callable that takes exit_context dict, returns bool.
                Context keys: direction, entry_price, current_price,
                unrealized_pnl_pips, hold_time, exit_reason.
        """
        self._exit_approval_fn = fn

    def _request_exit_approval(self, position: tuple, price: float, idx: int, reason: str) -> bool:
        """Check HITL approval for an exit if enabled.

        Returns True if exit is approved (or HITL is disabled).
        """
        if not self.config.human_exit_approval or self._exit_approval_fn is None:
            return True

        cfg = self.config
        direction, entry_price, entry_idx = position
        pnl_pips = (price - entry_price) * direction / cfg.pip_value
        hold_time = idx - entry_idx

        context = {
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": round(entry_price, 2),
            "current_price": round(price, 2),
            "unrealized_pnl_pips": round(pnl_pips, 1),
            "hold_time": hold_time,
            "exit_reason": reason,
        }
        return self._exit_approval_fn(context)

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
    ) -> BacktestResult:
        """Run backtest with given signals.

        Args:
            prices: Close prices (n_steps,).
            signals: Action signals (n_steps,) — 0=sell, 1=hold, 2=buy.

        Returns:
            BacktestResult with trades, equity curve, and metrics.
        """
        cfg = self.config
        balance = cfg.initial_balance
        position = None  # (direction, entry_price, entry_idx)
        trades = []
        equity = [balance]

        for i in range(len(prices)):
            signal = int(signals[i])
            price = float(prices[i])

            # Force close if max hold time exceeded
            if position and (i - position[2]) >= cfg.max_position_time:
                if self._request_exit_approval(position, price, i, "max_time"):
                    pnl = self._close(position, price, i, "max_time")
                    trades.append(pnl)
                    balance += pnl.pnl_usd
                    position = None
                # else: HITL vetoed — keep position open past max time

            # Execute signal
            if signal == 2 and position is None:  # Buy
                spread_cost = cfg.spread_pips * cfg.pip_value * 0.5
                slippage = np.random.uniform(0, cfg.slippage_pips * cfg.pip_value)
                entry = price + spread_cost + slippage
                position = (1, entry, i)  # long

            elif signal == 0 and position is None:  # Sell (short)
                spread_cost = cfg.spread_pips * cfg.pip_value * 0.5
                slippage = np.random.uniform(0, cfg.slippage_pips * cfg.pip_value)
                entry = price - spread_cost - slippage
                position = (-1, entry, i)  # short

            elif signal == 2 and position and position[0] == -1:  # Close short + open long
                if self._request_exit_approval(position, price, i, "signal_reverse"):
                    pnl = self._close(position, price, i, "signal_reverse")
                    trades.append(pnl)
                    balance += pnl.pnl_usd
                    entry = price + cfg.spread_pips * cfg.pip_value * 0.5
                    position = (1, entry, i)
                # else: HITL vetoed — keep short, skip buy entry

            elif signal == 0 and position and position[0] == 1:  # Close long + open short
                if self._request_exit_approval(position, price, i, "signal_reverse"):
                    pnl = self._close(position, price, i, "signal_reverse")
                    trades.append(pnl)
                    balance += pnl.pnl_usd
                    entry = price - cfg.spread_pips * cfg.pip_value * 0.5
                    position = (-1, entry, i)
                # else: HITL vetoed — keep long, skip short entry

            equity.append(balance)

        # Close remaining position at end (no HITL for end-of-data)
        if position:
            pnl = self._close(position, prices[-1], len(prices) - 1, "end_of_data")
            trades.append(pnl)
            balance += pnl.pnl_usd
            equity.append(balance)

        result = BacktestResult(
            trades=trades,
            equity_curve=equity,
            initial_balance=cfg.initial_balance,
            final_balance=balance,
        )
        logger.info(f"\n{result.summary()}")
        return result

    def _close(self, position: tuple, price: float, idx: int, reason: str) -> Trade:
        """Close a position and return the trade record."""
        cfg = self.config
        direction, entry_price, entry_idx = position

        spread_cost = cfg.spread_pips * cfg.pip_value * 0.5
        slippage = np.random.uniform(0, cfg.slippage_pips * cfg.pip_value)
        exit_price = price - (spread_cost + slippage) * direction

        pnl_pips = (exit_price - entry_price) * direction / cfg.pip_value
        pnl_usd = pnl_pips * cfg.pip_usd_per_lot * (cfg.lot_size / 0.01) - cfg.commission_per_lot

        return Trade(
            entry_idx=entry_idx,
            exit_idx=idx,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pips=pnl_pips,
            pnl_usd=pnl_usd,
            hold_time=idx - entry_idx,
            exit_reason=reason,
        )
