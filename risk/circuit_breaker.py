"""
Portfolio Circuit Breaker.

Tracks the portfolio's peak value. When drawdown from peak
exceeds max_drawdown (default -18%), all trading halts and
the system moves to cash.

The breaker auto-resets only when portfolio reaches a new all-time high.

State is persisted to disk so the drawdown tracking survives daily restarts.
"""

from __future__ import annotations
from pathlib import Path


class CircuitBreaker:
    def __init__(self, max_drawdown: float = 0.18):
        """
        Args:
            max_drawdown: drawdown threshold that triggers the halt (default 0.18 = -18%)
        """
        self.max_drawdown = max_drawdown
        self.peak_value: float | None = None
        self.triggered: bool = False
        self._trigger_value: float | None = None

    def update(self, portfolio_value: float) -> bool:
        """
        Record the current portfolio value and check the circuit breaker.

        Returns:
            True  — trading may continue
            False — breaker is triggered, halt all new trades
        """
        if self.peak_value is None:
            self.peak_value = portfolio_value

        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            if self.triggered:
                print(
                    f"Circuit breaker RESET — new portfolio high: "
                    f"${portfolio_value:,.2f}"
                )
                self.triggered = False
                self._trigger_value = None

        drawdown = (portfolio_value - self.peak_value) / self.peak_value

        if drawdown <= -self.max_drawdown and not self.triggered:
            self.triggered = True
            self._trigger_value = portfolio_value
            print(
                f"\n{'='*50}\n"
                f"CIRCUIT BREAKER TRIGGERED\n"
                f"  Drawdown:  {drawdown:.1%} (threshold: -{self.max_drawdown:.0%})\n"
                f"  Peak:      ${self.peak_value:,.2f}\n"
                f"  Current:   ${portfolio_value:,.2f}\n"
                f"  Action:    Halting all trades → move to cash\n"
                f"{'='*50}\n"
            )

        return not self.triggered

    @property
    def current_drawdown(self) -> float:
        if self.peak_value is None or self.peak_value == 0:
            return 0.0
        val = self._trigger_value or self.peak_value
        return (val - self.peak_value) / self.peak_value

    def status(self) -> dict:
        return {
            "triggered": self.triggered,
            "peak_value": self.peak_value,
            "trigger_value": self._trigger_value,
            "max_drawdown_threshold": f"-{self.max_drawdown:.0%}",
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save_state(self, path: Path) -> None:
        """Persist peak/trigger state to JSON so drawdown tracking survives restarts."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "peak_value":    self.peak_value,
            "triggered":     self.triggered,
            "trigger_value": self._trigger_value,
        }))

    @classmethod
    def load_state(cls, path: Path, max_drawdown: float = 0.18) -> "CircuitBreaker":
        """Load persisted state.  Returns a fresh breaker if no state file exists."""
        import json
        cb = cls(max_drawdown=max_drawdown)
        if path.exists():
            try:
                state = json.loads(path.read_text())
                cb.peak_value      = state.get("peak_value")
                cb.triggered       = state.get("triggered", False)
                cb._trigger_value  = state.get("trigger_value")
            except Exception:
                pass  # corrupted state — start fresh
        return cb
