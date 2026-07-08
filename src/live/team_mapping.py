"""Map external (API-Football) team names to the model's training vocabulary.

The CatBoost model learned `HomeTeam`/`AwayTeam` as raw strings using
football-data.co.uk's idiosyncratic naming (e.g. "Ath Madrid", "Espanol",
"Sociedad"). A live provider uses different names ("Atletico Madrid",
"Espanyol", "Real Sociedad"), so we translate before building features.

An unmapped team is returned unchanged and flagged by the caller: CatBoost
treats an unseen category gracefully (it just carries less signal).
"""

import unicodedata

# The 37 team strings the model was trained on (La Liga 2008/09–2018/19).
# Source: sorted(set(X_train.HomeTeam) | set(X_train.AwayTeam)).
TRAINING_TEAMS = {
    "Alaves", "Almeria", "Ath Bilbao", "Ath Madrid", "Barcelona", "Betis",
    "Celta", "Cordoba", "Eibar", "Elche", "Espanol", "Getafe", "Girona",
    "Granada", "Hercules", "Huesca", "La Coruna", "Las Palmas", "Leganes",
    "Levante", "Malaga", "Mallorca", "Numancia", "Osasuna", "Real Madrid",
    "Recreativo", "Santander", "Sevilla", "Sociedad", "Sp Gijon", "Tenerife",
    "Valencia", "Valladolid", "Vallecano", "Villarreal", "Xerez", "Zaragoza",
}


def _normalize(name: str) -> str:
    """Lowercase, strip accents and collapse whitespace for robust matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(no_accents.lower().replace(".", " ").split())


# Explicit overrides keyed by the *normalized* external name. Only names that do
# not already match a training team after normalization need an entry here.
_EXPLICIT = {
    "atletico madrid": "Ath Madrid",
    "atletico de madrid": "Ath Madrid",
    "atletico": "Ath Madrid",
    "athletic club": "Ath Bilbao",
    "athletic bilbao": "Ath Bilbao",
    "real sociedad": "Sociedad",
    "real betis": "Betis",
    "espanyol": "Espanol",
    "rcd espanyol": "Espanol",
    "celta vigo": "Celta",
    "rc celta": "Celta",
    "rc celta de vigo": "Celta",
    "rayo vallecano": "Vallecano",
    "deportivo alaves": "Alaves",
    "real valladolid": "Valladolid",
    "deportivo la coruna": "La Coruna",
    "deportivo": "La Coruna",
    "granada cf": "Granada",
    "ud almeria": "Almeria",
    "ud las palmas": "Las Palmas",
    "cd leganes": "Leganes",
    "sd huesca": "Huesca",
    "sd eibar": "Eibar",
    "levante ud": "Levante",
    "getafe cf": "Getafe",
    "villarreal cf": "Villarreal",
    "valencia cf": "Valencia",
    "sevilla fc": "Sevilla",
    "fc barcelona": "Barcelona",
    "ca osasuna": "Osasuna",
    "rcd mallorca": "Mallorca",
    "elche cf": "Elche",
    "girona fc": "Girona",
    "malaga cf": "Malaga",
    "real zaragoza": "Zaragoza",
    "sporting gijon": "Sp Gijon",
    "real sporting": "Sp Gijon",
    "real sporting de gijon": "Sp Gijon",
    "racing santander": "Santander",
    "racing de santander": "Santander",
    "recreativo huelva": "Recreativo",
    "recreativo de huelva": "Recreativo",
    "cordoba cf": "Cordoba",
    "cd tenerife": "Tenerife",
}

# Precompute normalized -> canonical for the training vocabulary itself.
_NORMALIZED_TRAINING = {_normalize(t): t for t in TRAINING_TEAMS}


def map_team(name: str) -> tuple[str, bool]:
    """Translate an external team name to the training vocabulary.

    Returns (mapped_name, matched) where `matched` is False when the team is not
    in the training vocabulary (the original name is returned in that case).
    """
    norm = _normalize(name)
    if norm in _EXPLICIT:
        return _EXPLICIT[norm], True
    if norm in _NORMALIZED_TRAINING:
        return _NORMALIZED_TRAINING[norm], True
    return name, False
