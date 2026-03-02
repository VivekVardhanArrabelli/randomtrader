"""AI-driven autonomous options trader.

An LLM reads market news, analyzes conditions, determines option trades,
and executes them autonomously. Each trade risks at most 40% of the portfolio.
"""

from . import config

__all__ = ["config"]
