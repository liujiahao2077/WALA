"""Action head modules used by WALA."""

from .mlp_action_head import L1RegressionActionHead, get_action_model

__all__ = ["L1RegressionActionHead", "get_action_model"]
