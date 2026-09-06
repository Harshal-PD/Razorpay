"""Defense-only policy: never silent deny / cancel / region ban."""

from __future__ import annotations

import pytest

from src.policy import ACTIONS, FORBIDDEN_ACTIONS, decide, medium_threshold


def test_high_risk_requires_prepaid_never_denied():
    out = decide(0.99, t_med=0.2, t_star=0.4, features_ok=True)
    assert out["risk_tier"] == "high"
    assert out["recommended_action"] == ACTIONS["high"]
    assert "prepaid" in out["recommended_action"]
    assert out["policy"] == "never_silent_deny"
    assert out["recommended_action"] not in FORBIDDEN_ACTIONS
    assert "denied" not in out["recommended_action"]
    assert "ban" not in out["recommended_action"]


def test_low_allows_cod():
    out = decide(0.05, t_med=0.2, t_star=0.5, features_ok=True)
    assert out["risk_tier"] == "low"
    assert out["recommended_action"] == "proceed_cod_allowed"


def test_medium_is_verify_before_dispatch():
    out = decide(0.3, t_med=0.2, t_star=0.5, features_ok=True)
    assert out["risk_tier"] == "medium"
    assert out["recommended_action"] == "verify_before_dispatch"


def test_insufficient_history_is_conservative_medium_not_deny():
    out = decide(None, t_med=0.2, t_star=0.5, features_ok=False)
    assert out["risk_tier"] == "medium"
    assert out["reason"] == "insufficient_history"
    assert out["recommended_action"] == "verify_before_dispatch"
    assert out["policy"] == "never_silent_deny"
    assert out["recommended_action"] != "denied"


def test_grid_never_emits_forbidden_actions():
    t_star = 0.4
    t_med = medium_threshold(t_star)
    for p in [i / 20 for i in range(21)]:
        out = decide(p, t_med=t_med, t_star=t_star)
        action = out["recommended_action"]
        for bad in FORBIDDEN_ACTIONS:
            assert bad not in action, f"{bad} leaked at p={p}: {action}"
        assert out["risk_tier"] in {"low", "medium", "high"}
        assert out["policy"] == "never_silent_deny"


def test_high_action_is_not_region_ban():
    out = decide(1.0, t_med=0.1, t_star=0.2)
    assert "pin" not in out["recommended_action"]
    assert out["recommended_action"] == "require_prepaid_or_address_reverify"
