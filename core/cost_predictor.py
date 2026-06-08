# core/cost_predictor.py

COST_BUCKETS = [
    {
        "bucket":       "Cosmetic Fix",
        "range":        "₹1,000 – ₹3,000",
        "min":          1000,
        "max":          3000,
        "score_min":    0.0,
        "score_max":    1.5,
        "description":  "Minor surface issues. Polish, touch-up paint or PDR sufficient.",
        "priority":     "Low",
    },
    {
        "bucket":       "Standard Repair",
        "range":        "₹3,000 – ₹6,000",
        "min":          3000,
        "max":          6000,
        "score_min":    1.5,
        "score_max":    3.5,
        "description":  "Moderate damage. Panel repair and repaint required.",
        "priority":     "Medium",
    },
    {
        "bucket":       "Major Repair",
        "range":        "₹6,000 – ₹12,000",
        "min":          6000,
        "max":          12000,
        "score_min":    3.5,
        "score_max":    6.0,
        "description":  "Significant structural or cosmetic damage. Full panel work needed.",
        "priority":     "High",
    },
    {
        "bucket":       "Panel Replacement",
        "range":        "₹12,000 – ₹25,000",
        "min":          12000,
        "max":          25000,
        "score_min":    6.0,
        "score_max":    8.0,
        "description":  "Severe damage. Replacement more cost-effective than repair.",
        "priority":     "Critical",
    },
    {
        "bucket":       "Insurance / Write-off Risk",
        "range":        "₹25,000+",
        "min":          25000,
        "max":          None,
        "score_min":    8.0,
        "score_max":    10.1,
        "description":  "Extensive multi-panel or structural damage. Flag for insurance review.",
        "priority":     "Critical",
    },
]


def predict_cost(damage_score: float, panel: str = None) -> dict:
    """
    Maps a damage score (0-10) to a cost bucket with full metadata.
    """
    for bucket in COST_BUCKETS:
        if bucket["score_min"] <= damage_score < bucket["score_max"]:
            result = {
                "damage_score":     damage_score,
                "cost_bucket":      bucket["bucket"],
                "cost_range":       bucket["range"],
                "cost_min":         bucket["min"],
                "cost_max":         bucket["max"],
                "priority":         bucket["priority"],
                "description":      bucket["description"],
            }
            if panel:
                result["panel"] = panel
            return result

    # Fallback — should never hit but safety net
    return {
        "damage_score":     damage_score,
        "cost_bucket":      "Unknown",
        "cost_range":       "Manual Review Required",
        "cost_min":         0,
        "cost_max":         None,
        "priority":         "Review",
        "description":      "Score outside expected range. Manual inspection recommended.",
    }


def predict_cost_from_engine_output(engine_result: dict) -> dict:
    """
    Takes output dict from severity_engine.compute_damage_score()
    and returns full combined result.
    """
    score = engine_result.get("damage_score", 0)
    panel = engine_result.get("panel", "")
    cost = predict_cost(score, panel)

    return {**engine_result, **cost}


# Quick sanity test
if __name__ == "__main__":
    from core.severity_engine import score_from_raw

    test_cases = [
        ("Scratch-Minor",                               "Left Running Board"),
        ("Scratch-Minor, Dent-Major",                   "Front Bumper"),
        ("Crack-Major, Tear-Major, Replaced",           "Windshield"),
        ("Dent-Major, Misalignment-Major, Accident-Damage", "Bonnet"),
        ("Replaced, Broken, Deformed, Accident-Damage", "Front Bumper"),
    ]

    print(f"\n{'Panel':<25} {'Score':>6}  {'Bucket':<25} {'Range':<22} {'Priority'}")
    print("-" * 95)

    for issues, panel in test_cases:
        engine_out = score_from_raw(issues, panel)
        result = predict_cost_from_engine_output(engine_out)
        print(
            f"{result['panel']:<25} "
            f"{result['damage_score']:>6}  "
            f"{result['cost_bucket']:<25} "
            f"{result['cost_range']:<22} "
            f"{result['priority']}"
        )