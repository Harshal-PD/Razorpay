"""Defense-only action policy.

Never silent cancel, never region ban, never auto-block.
PIN-tier is for verification routing, not refusal.
"""

from __future__ import annotations

from typing import Any

DEFAULT_T_MED = 0.18
DEFAULT_T_STAR = 0.42

POLICY_NAME = "never_silent_deny"

ACTIONS = {
    "low": "proceed_cod_allowed",
    "medium": "verify_before_dispatch",
    "high": "require_prepaid_or_address_reverify",
}

# Hard deny vocabulary that must never appear in recommended_action
FORBIDDEN_ACTIONS = frozenset(
    {
        "denied",
        "deny",
        "silent_cancel",
        "cancel",
        "auto_block",
        "block",
        "region_ban",
        "ban",
        "do_not_ship",
        "refuse",
    }
)


def medium_threshold(t_star: float, t_med: float | None = None) -> float:
    if t_med is not None:
        return float(min(t_med, t_star * 0.999))
    return float(max(0.08, min(t_star * 0.50, t_star - 0.06)))


def decide(
    probability: float | None,
    t_med: float,
    t_star: float,
    *,
    features_ok: bool = True,
    top_factors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map probability → tier + action. Missing features → conservative medium."""
    if t_star <= t_med:
        t_med = max(0.05, t_star * 0.5)

    if (not features_ok) or probability is None:
        action = ACTIONS["medium"]
        out = {
            "rto_probability": None if probability is None else float(probability),
            "risk_tier": "medium",
            "recommended_action": action,
            "reason": "insufficient_history",
            "top_factors": top_factors or [],
            "policy": POLICY_NAME,
        }
        _assert_never_deny(out)
        return out

    p = float(probability)
    if p >= t_star:
        tier = "high"
    elif p >= t_med:
        tier = "medium"
    else:
        tier = "low"

    out = {
        "rto_probability": p,
        "risk_tier": tier,
        "recommended_action": ACTIONS[tier],
        "reason": "scored",
        "top_factors": top_factors or [],
        "policy": POLICY_NAME,
    }
    _assert_never_deny(out)
    return out


def _assert_never_deny(payload: dict[str, Any]) -> None:
    action = str(payload.get("recommended_action", "")).lower()
    for bad in FORBIDDEN_ACTIONS:
        if bad in action.replace("-", "_").split("_") or action == bad:
            raise RuntimeError(f"Policy violation: action '{action}' is not defense-only")
        if bad in action.replace("-", "_"):
            # extra catch for concatenated forms
            if bad in {"denied", "deny", "region_ban", "silent_cancel", "auto_block"}:
                raise RuntimeError(f"Policy violation: forbidden token in action '{action}'")
    if payload.get("risk_tier") == "denied":
        raise RuntimeError("Policy violation: denied tier")


def fairness_note(fp_by_pin: dict[str, dict[str, float]]) -> str:
    if not fp_by_pin:
        return "No PIN-tier FP breakdown available."
    rates = {k: v.get("fp_rate", float("nan")) for k, v in fp_by_pin.items()}
    if "tier3" in rates and "metro" in rates:
        t3, m = rates["tier3"], rates["metro"]
        if t3 == t3 and m == m and m > 0 and t3 > 1.5 * m:
            return (
                f"Fairness call-out: tier3 FP rate ({t3:.3f}) is much higher than metro "
                f"({m:.3f}). PIN is for extra verification routing, never a ship-ban."
            )
    return (
        "PIN-tier scores can proxy caste/religion/class geography. "
        "Use extra verification, never 'we don't ship to this PIN'."
    )
