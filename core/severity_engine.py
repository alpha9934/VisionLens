# core/severity_engine.py

ISSUE_SEVERITY_MAP = {
    # Low severity — surface level, cheap fix
    "Scratch-Minor":        1,
    "Dent-Minor":           1,
    "Crack-Minor":          1,
    "Fade-Minor":           1,
    "Rust-Minor":           1,
    "Chip-Minor":           1,

    # Medium severity — needs proper repair
    "Scratch-Major":        2,
    "Dent-Major":           2,
    "Crack-Major":          2,
    "Fade-Major":           2,
    "Rust-Major":           2,
    "Chip-Major":           2,
    "Tear-Minor":           2,
    "Misalignment-Minor":   2,
    "Paint-Peel-Minor":     2,

    # High severity — structural or full replacement risk
    "Tear-Major":           3,
    "Misalignment-Major":   3,
    "Paint-Peel-Major":     3,
    "Replaced":             3,
    "Broken":               3,
    "Missing":              3,
    "Deformed":             3,
    "Accident-Damage":      3,
}

PANEL_WEIGHT = {
    # Structural panels — higher impact on resale
    "Front Bumper":         1.3,
    "Rear Bumper":          1.2,
    "Bonnet":               1.3,
    "Dickey":               1.1,
    "Front Left Door":      1.2,
    "Front Right Door":     1.2,
    "Rear Left Door":       1.1,
    "Rear Right Door":      1.1,
    "Left Quarter Panel":   1.2,
    "Right Quarter Panel":  1.2,
    "Left Running Board":   1.0,
    "Right Running Board":  1.0,
    "Roof":                 1.3,
    "Windshield":           1.4,
}


def parse_issues(raw_issue_string: str) -> list[str]:
    """
    Converts raw comma-separated issue string from dataset into clean list.
    Example: "Scratch-Minor, Dent-Major" → ["Scratch-Minor", "Dent-Major"]
    """
    if not raw_issue_string or str(raw_issue_string).strip().lower() == "nan":
        return []
    return [i.strip() for i in str(raw_issue_string).split(",") if i.strip()]


def compute_damage_score(issues: list[str], panel: str) -> dict:
    """
    Core scoring function.
    Returns damage score (0-10), severity breakdown, and unknown issues.
    """
    raw_score = 0
    matched = []
    unmatched = []

    for issue in issues:
        weight = ISSUE_SEVERITY_MAP.get(issue)
        if weight:
            raw_score += weight
            matched.append({"issue": issue, "weight": weight})
        else:
            # Fallback — unknown issue gets medium weight
            raw_score += 1
            unmatched.append(issue)

    # Apply panel structural weight
    panel_multiplier = PANEL_WEIGHT.get(panel, 1.0)
    weighted_score = raw_score * panel_multiplier

    # Normalize to 0-10 scale
    # Max possible raw = 10 issues × 3 (max weight) × 1.4 (max panel weight) = 42
    MAX_POSSIBLE = 42.0
    normalized = round(min((weighted_score / MAX_POSSIBLE) * 10, 10.0), 2)

    return {
        "panel": panel,
        "issues_parsed": issues,
        "matched_issues": matched,
        "unmatched_issues": unmatched,
        "raw_score": raw_score,
        "panel_multiplier": panel_multiplier,
        "damage_score": normalized,
    }


def score_from_raw(raw_issue_string: str, panel: str) -> dict:
    """
    Convenience wrapper — takes raw string directly from dataset row.
    """
    issues = parse_issues(raw_issue_string)
    return compute_damage_score(issues, panel)


# Quick sanity test
if __name__ == "__main__":
    test_cases = [
        ("Scratch-Minor, Dent-Major", "Front Bumper"),
        ("Crack-Major, Tear-Major, Replaced", "Windshield"),
        ("Fade-Minor", "Left Running Board"),
        ("Dent-Major, Misalignment-Major, Accident-Damage", "Bonnet"),
    ]

    for issues, panel in test_cases:
        result = score_from_raw(issues, panel)
        print(f"\nPanel: {panel}")
        print(f"Issues: {issues}")
        print(f"Damage Score: {result['damage_score']} / 10")
        print(f"Raw Score: {result['raw_score']} × {result['panel_multiplier']} multiplier")