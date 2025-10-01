import argparse
import pandas as pd
from typing import Dict, Any, List

DECISION_ACCEPTED = "ACCEPTED"
DECISION_IN_REVIEW = "IN_REVIEW"
DECISION_REJECTED = "REJECTED"

DEFAULT_CONFIG = {
    "amount_thresholds": {
        "digital": 2500,
        "physical": 6000,
        "subscription": 1500,
        "_default": 4000
    },
    "latency_ms_extreme": 2500,
    "chargeback_hard_block": 2,
    "score_weights": {
        "ip_risk": {"low": 0, "medium": 2, "high": 4},
        "email_risk": {"low": 0, "medium": 1, "high": 3, "new_domain": 2},
        "device_fingerprint_risk": {"low": 0, "medium": 2, "high": 4},
        "user_reputation": {"trusted": -2, "recurrent": -1, "new": 0, "high_risk": 4},
        "night_hour": 1,
        "geo_mismatch": 2,
        "high_amount": 2,
        "latency_extreme": 2,
        "new_user_high_amount": 2,
    },
    "score_to_decision": {
        "reject_at": 10,
        "review_at": 4
    }
}

# Optional: override thresholds via environment variables (for CI/CD / canary tuning)
try:
    import os as _os
    _rej = _os.getenv("REJECT_AT")
    _rev = _os.getenv("REVIEW_AT")
    if _rej is not None:
        DEFAULT_CONFIG["score_to_decision"]["reject_at"] = int(_rej)
    if _rev is not None:
        DEFAULT_CONFIG["score_to_decision"]["review_at"] = int(_rev)
except Exception:
    pass

def is_night(hour: int) -> bool:
    return hour >= 22 or hour <= 5

def high_amount(amount: float, product_type: str, thresholds: Dict[str, Any]) -> bool:
    t = thresholds.get(product_type, thresholds.get("_default"))
    return amount >= t


def check_hard_block(row: pd.Series, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Check for hard block conditions that immediately reject the transaction."""
    chargeback_count = int(row.get("chargeback_count", 0))
    ip_risk = str(row.get("ip_risk", "low")).lower()

    if chargeback_count >= cfg["chargeback_hard_block"] and ip_risk == "high":
        return {
            "decision": DECISION_REJECTED,
            "risk_score": 100,
            "reasons": "hard_block:chargebacks>=2+ip_high"
        }
    return None


def calculate_categorical_risks(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Calculate risk scores for categorical fields."""
    score = 0
    reasons = []

    categorical_fields = [
        ("ip_risk", cfg["score_weights"]["ip_risk"]),
        ("email_risk", cfg["score_weights"]["email_risk"]),
        ("device_fingerprint_risk", cfg["score_weights"]["device_fingerprint_risk"])
    ]

    for field, mapping in categorical_fields:
        val = str(row.get(field, "low")).lower()
        add = mapping.get(val, 0)
        score += add
        if add:
            reasons.append(f"{field}:{val}(+{add})")

    return score, reasons


def calculate_reputation_score(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, List[str], str]:
    """Calculate reputation-based risk score."""
    rep = str(row.get("user_reputation", "new")).lower()
    rep_add = cfg["score_weights"]["user_reputation"].get(rep, 0)
    reasons = []

    if rep_add:
        sign = '+' if rep_add >= 0 else ''
        reasons.append(f"user_reputation:{rep}({sign}{rep_add})")

    return rep_add, reasons, rep


def calculate_time_and_geo_risks(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Calculate risks based on time and geographical factors."""
    score = 0
    reasons = []

    # Night hour risk
    hr = int(row.get("hour", 12))
    if is_night(hr):
        add = cfg["score_weights"]["night_hour"]
        score += add
        reasons.append(f"night_hour:{hr}(+{add})")

    # Geo mismatch risk
    bin_c = str(row.get("bin_country", "")).upper()
    ip_c = str(row.get("ip_country", "")).upper()
    if bin_c and ip_c and bin_c != ip_c:
        add = cfg["score_weights"]["geo_mismatch"]
        score += add
        reasons.append(f"geo_mismatch:{bin_c}!={ip_c}(+{add})")

    return score, reasons


def calculate_amount_and_latency_risks(row: pd.Series, cfg: Dict[str, Any], rep: str) -> Tuple[int, List[str]]:
    """Calculate risks based on transaction amount and latency."""
    score = 0
    reasons = []

    # High amount risk
    amount = float(row.get("amount_mxn", 0.0))
    ptype = str(row.get("product_type", "_default")).lower()
    if high_amount(amount, ptype, cfg["amount_thresholds"]):
        add = cfg["score_weights"]["high_amount"]
        score += add
        reasons.append(f"high_amount:{ptype}:{amount}(+{add})")

        if rep == "new":
            add2 = cfg["score_weights"]["new_user_high_amount"]
            score += add2
            reasons.append(f"new_user_high_amount(+{add2})")

    # Extreme latency risk
    lat = int(row.get("latency_ms", 0))
    if lat >= cfg["latency_ms_extreme"]:
        add = cfg["score_weights"]["latency_extreme"]
        score += add
        reasons.append(f"latency_extreme:{lat}ms(+{add})")

    return score, reasons


def apply_frequency_buffer(score: int, rep: str, row: pd.Series) -> Tuple[int, List[str]]:
    """Apply frequency buffer for trusted/recurrent users."""
    reasons = []
    freq = int(row.get("customer_txn_30d", 0))

    if rep in ("recurrent", "trusted") and freq >= 3 and score > 0:
        score -= 1
        reasons.append("frequency_buffer(-1)")

    return score, reasons


def determine_decision(score: int, cfg: Dict[str, Any]) -> str:
    """Determine final decision based on risk score."""
    if score >= cfg["score_to_decision"]["reject_at"]:
        return DECISION_REJECTED
    elif score >= cfg["score_to_decision"]["review_at"]:
        return DECISION_IN_REVIEW
    else:
        return DECISION_ACCEPTED


def assess_row(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to assess transaction risk."""
    # Check for hard block conditions first
    hard_block_result = check_hard_block(row, cfg)
    if hard_block_result:
        return hard_block_result

    # Calculate risk scores from different categories
    score = 0
    all_reasons = []

    # Categorical risks
    cat_score, cat_reasons = calculate_categorical_risks(row, cfg)
    score += cat_score
    all_reasons.extend(cat_reasons)

    # Reputation score
    rep_score, rep_reasons, rep = calculate_reputation_score(row, cfg)
    score += rep_score
    all_reasons.extend(rep_reasons)

    # Time and geo risks
    time_geo_score, time_geo_reasons = calculate_time_and_geo_risks(row, cfg)
    score += time_geo_score
    all_reasons.extend(time_geo_reasons)

    # Amount and latency risks
    amount_lat_score, amount_lat_reasons = calculate_amount_and_latency_risks(row, cfg, rep)
    score += amount_lat_score
    all_reasons.extend(amount_lat_reasons)

    # Apply frequency buffer
    score, buffer_reasons = apply_frequency_buffer(score, rep, row)
    all_reasons.extend(buffer_reasons)

    # Determine final decision
    decision = determine_decision(score, cfg)

    return {
        "decision": decision,
        "risk_score": int(score),
        "reasons": ";".join(all_reasons)
    }
def run(input_csv: str, output_csv: str, config: Dict[str, Any] = None) -> pd.DataFrame:
    cfg = config or DEFAULT_CONFIG
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        res = assess_row(row, cfg)
        results.append(res)
    out = df.copy()
    out["decision"] = [r["decision"] for r in results]
    out["risk_score"] = [r["risk_score"] for r in results]
    out["reasons"] = [r["reasons"] for r in results]
    out.to_csv(output_csv, index=False)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, default="transactions_examples.csv", help="Path to input CSV")
    ap.add_argument("--output", required=False, default="decisions.csv", help="Path to output CSV")
    args = ap.parse_args()
    out = run(args.input, args.output)
    print(out.head().to_string(index=False))

if __name__ == "__main__":
    main()
