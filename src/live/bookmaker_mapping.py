"""Map external bookmakers to the 5 odds slots the model expects.

The `odds_form_teams` feature set uses five bookmaker triplets:
    BW  = Bwin
    IW  = Interwetten
    WH  = William Hill
    VC  = (BetVictor / Victor Chandler)
    PSC = Pinnacle "closing" odds
Live providers expose a different, varying set of bookmakers, so we match by
name. Pinnacle's pre-match odds stand in for the PSC (closing) slot — a
documented approximation; the model consumes *normalized* implied probabilities,
which are highly correlated across bookmakers.

Any slot we cannot fill from the live data is backfilled by the feature builder
with the market consensus (mean of the available slots) — never left at 0.0,
which would be a catastrophic value for a normalized-probability column.
"""

# Order matters: it is the bookmaker order within the odds_form_teams feature set.
MODEL_BOOKMAKERS = ["BW", "IW", "WH", "VC", "PSC"]

# Substring patterns (matched case-insensitively against the provider's bookmaker
# name) -> model slot. First match wins.
_NAME_PATTERNS = [
    ("bwin", "BW"),
    ("interwetten", "IW"),
    ("william hill", "WH"),
    ("betvictor", "VC"),
    ("bet victor", "VC"),
    ("victor chandler", "VC"),
    ("pinnacle", "PSC"),
]


def bookmaker_to_slot(name: str) -> str | None:
    """Return the model slot ('BW'|'IW'|'WH'|'VC'|'PSC') for a bookmaker, or None."""
    if not name:
        return None
    low = name.lower()
    for pattern, slot in _NAME_PATTERNS:
        if pattern in low:
            return slot
    return None
