
"""
For1Sight V2.8 - Future Race Prediction Engine
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "starting_grid",
    "driver_prior_avg_finish_3",
    "driver_prior_avg_points_3",
    "constructor_prior_avg_finish_3",
    "constructor_prior_avg_points_3",
    "driver_code_norm",
    "constructor",
    "event_norm",
]

NUMERIC_FEATURES = FEATURE_COLUMNS[:5]
CATEGORICAL_FEATURES = FEATURE_COLUMNS[5:]

TARGET_SEASON_ROSTER = 2026


def normalize_event_name(value):
    value = str(value or "").lower().strip()
    value = re.sub(r"\bgrand\s*prix\b", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def validate_warehouse(df):
    required = {
        "year", "round", "event_date", "event_name",
        "event_norm", "event_id", "driver_code",
        "driver_code_norm", "constructor", "starting_grid",
        "finishing_position", "race_points"
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Missing warehouse columns: " + ", ".join(missing)
        )

    if len(FEATURE_COLUMNS) != 8:
        raise AssertionError("V2.8 must use exactly eight features.")


def _prepare_history(df, target_year):

    validate_warehouse(df)

    work = df.copy()

    work["year_num"] = pd.to_numeric(
        work["year"], errors="coerce"
    )

    work["round_num"] = pd.to_numeric(
        work["round"], errors="coerce"
    )

    work["event_date_sort"] = pd.to_datetime(
        work["event_date"], errors="coerce"
    )

    work["finish_num"] = pd.to_numeric(
        work["finishing_position"], errors="coerce"
    )

    work["points_num"] = pd.to_numeric(
        work["race_points"], errors="coerce"
    )

    work["starting_grid"] = pd.to_numeric(
        work["starting_grid"], errors="coerce"
    )

    work["driver_code_norm"] = (
        work["driver_code_norm"]
        .astype(str).str.upper().str.strip()
    )

    work["constructor"] = (
        work["constructor"].fillna("Unknown").astype(str)
    )

    work["event_norm"] = (
        work["event_norm"].fillna("").astype(str)
    )

    valid = (
        (work["year_num"] < int(target_year))
        & work["event_date_sort"].notna()
        & work["finish_num"].notna()
        & work["points_num"].notna()
        & work["starting_grid"].notna()
        & work["driver_code_norm"].ne("")
        & work["driver_code_norm"].ne("NAN")
    )

    if "event_prediction_ready" in work.columns:
        valid &= work["event_prediction_ready"].fillna(False).astype(bool)

    if "model_training_eligible" in work.columns:
        valid &= work["model_training_eligible"].fillna(False).astype(bool)

    if "target" in work.columns:
        valid &= pd.to_numeric(
            work["target"], errors="coerce"
        ).isin([0, 1])

    history = work.loc[valid].copy()

    if history.empty:
        raise ValueError(
            f"No historical records found before {target_year}."
        )

    history = history.sort_values(
        [
            "event_date_sort",
            "year_num",
            "round_num",
            "event_id",
            "driver_code_norm"
        ],
        kind="mergesort"
    )

    duplicates = history.duplicated(
        ["event_id", "driver_code_norm"],
        keep=False
    )

    if duplicates.any():
        raise ValueError(
            "Duplicate driver-event records detected in warehouse."
        )

    return history


def get_race_catalog(df):

    validate_warehouse(df)

    catalog = df[
        ["event_name", "event_norm", "year", "round"]
    ].copy()

    catalog["year"] = pd.to_numeric(
        catalog["year"], errors="coerce"
    )

    catalog["round"] = pd.to_numeric(
        catalog["round"], errors="coerce"
    )

    catalog = catalog.dropna(
        subset=["event_name", "event_norm", "year", "round"]
    )

    catalog = catalog.sort_values(
        ["year", "round"]
    )

    catalog = catalog.drop_duplicates(
        "event_norm", keep="last"
    )

    return catalog.sort_values(
        "event_name"
    ).reset_index(drop=True)


def get_current_driver_roster(
    df, roster_year=TARGET_SEASON_ROSTER
):

    validate_warehouse(df)

    work = df.copy()

    work["year_num"] = pd.to_numeric(
        work["year"], errors="coerce"
    )

    work["round_num"] = pd.to_numeric(
        work["round"], errors="coerce"
    )

    work["driver_code_norm"] = (
        work["driver_code_norm"]
        .astype(str).str.upper().str.strip()
    )

    season = work.loc[
        work["year_num"].eq(int(roster_year))
    ].dropna(
        subset=["round_num", "driver_code_norm"]
    )

    if season.empty:
        raise ValueError(
            f"No {roster_year} driver records available."
        )

    latest_round = season["round_num"].max()

    latest_event = season.loc[
        season["round_num"].eq(latest_round)
    ].copy()

    latest_event = latest_event.sort_values(
        "driver_code_norm"
    ).drop_duplicates(
        "driver_code_norm", keep="last"
    )

    latest_event["constructor"] = (
        latest_event["constructor"]
        .fillna("Unknown").astype(str)
    )

    if len(latest_event) != 22:
        raise ValueError(
            f"Expected 22 drivers but found {len(latest_event)} "
            f"in the latest {roster_year} event."
        )

    return latest_event.reset_index(drop=True), int(roster_year)


def _driver_recent_form(history, driver_code):

    rows = history.loc[
        history["driver_code_norm"].eq(driver_code)
    ].tail(3)

    if rows.empty:
        return np.nan, np.nan

    return (
        float(rows["finish_num"].mean()),
        float(rows["points_num"].mean())
    )


def _constructor_event_history(history):

    return (
        history.groupby(
            [
                "event_id",
                "year_num",
                "round_num",
                "event_date_sort",
                "constructor"
            ],
            dropna=False,
            sort=False
        )
        .agg(
            constructor_event_avg_finish=(
                "finish_num", "mean"
            ),
            constructor_event_avg_points=(
                "points_num", "mean"
            )
        )
        .reset_index()
        .sort_values(
            [
                "event_date_sort",
                "year_num",
                "round_num",
                "event_id",
                "constructor"
            ],
            kind="mergesort"
        )
    )


def _constructor_recent_form(
    constructor_events, constructor
):

    rows = constructor_events.loc[
        constructor_events["constructor"].astype(str).eq(
            str(constructor)
        )
    ].tail(3)

    if rows.empty:
        return np.nan, np.nan

    return (
        float(rows["constructor_event_avg_finish"].mean()),
        float(rows["constructor_event_avg_points"].mean())
    )


def get_roster_as_of(
    df,
    target_year,
    target_round=None,
    driver_codes=None
):

    """
    Calculate historical form using records strictly before
    the selected prediction season.

    target_round is retained for compatibility with the
    existing Streamlit page. V2.8 does not use it as a feature.
    """

    history = _prepare_history(df, target_year)

    if driver_codes is None:

        roster, _ = get_current_driver_roster(df)

        requested = roster[
            "driver_code_norm"
        ].astype(str).str.upper().tolist()

    else:

        requested = [
            str(code).upper().strip()
            for code in driver_codes
        ]

    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate driver codes detected.")

    if not requested:
        raise ValueError("No drivers supplied.")

    latest_roster, _ = get_current_driver_roster(df)

    constructor_lookup = (
        latest_roster
        .assign(
            driver_code_norm=latest_roster[
                "driver_code_norm"
            ].astype(str).str.upper()
        )
        .drop_duplicates(
            "driver_code_norm", keep="last"
        )
        .set_index("driver_code_norm")["constructor"]
        .to_dict()
    )

    constructor_events = _constructor_event_history(history)

    rows = []

    for code in requested:

        constructor = constructor_lookup.get(code)

        if not constructor:
            raise ValueError(
                f"Driver {code} is not in the fixed 2026 roster."
            )

        driver_finish, driver_points = (
            _driver_recent_form(history, code)
        )

        constructor_finish, constructor_points = (
            _constructor_recent_form(
                constructor_events, constructor
            )
        )

        rows.append({
            "driver_code_norm": code,
            "constructor": str(constructor),
            "driver_prior_avg_finish_3": driver_finish,
            "driver_prior_avg_points_3": driver_points,
            "constructor_prior_avg_finish_3": constructor_finish,
            "constructor_prior_avg_points_3": constructor_points
        })

    result = pd.DataFrame(rows)

    result.attrs["target_year"] = int(target_year)
    result.attrs["historical_records_used"] = len(history)
    result.attrs["historical_latest_year"] = int(
        history["year_num"].max()
    )

    return result


def _historical_score(roster):

    def percentile(series, ascending=True):

        values = pd.to_numeric(
            series, errors="coerce"
        )

        if values.notna().sum() == 0:
            return pd.Series(
                0.5, index=series.index
            )

        values = values.fillna(values.median())

        return values.rank(
            pct=True, ascending=ascending
        )

    driver_points = percentile(
        roster["driver_prior_avg_points_3"], True
    )

    driver_finish = percentile(
        roster["driver_prior_avg_finish_3"], False
    )

    team_points = percentile(
        roster["constructor_prior_avg_points_3"], True
    )

    team_finish = percentile(
        roster["constructor_prior_avg_finish_3"], False
    )

    return (
        0.35 * driver_points
        + 0.25 * driver_finish
        + 0.25 * team_points
        + 0.15 * team_finish
    )


def fill_remaining_by_history(roster, fixed_top):

    if len(fixed_top) != 5 or len(set(fixed_top)) != 5:
        raise ValueError(
            "Please choose five different drivers for P1-P5."
        )

    work = roster.copy()

    work["driver_code_norm"] = (
        work["driver_code_norm"].astype(str).str.upper()
    )

    codes = work["driver_code_norm"].tolist()

    fixed_top = [
        str(code).upper() for code in fixed_top
    ]

    if len(codes) != 22 or len(set(codes)) != 22:
        raise ValueError(
            "The driver roster must contain 22 unique drivers."
        )

    if not set(fixed_top).issubset(set(codes)):
        raise ValueError(
            "Selected driver is not in the 2026 roster."
        )

    remaining = work.loc[
        ~work["driver_code_norm"].isin(fixed_top)
    ].copy()

    remaining["historical_form_score"] = (
        _historical_score(remaining)
    )

    remaining = remaining.sort_values(
        [
            "historical_form_score",
            "driver_prior_avg_points_3"
        ],
        ascending=[False, False],
        na_position="last",
        kind="mergesort"
    )

    return (
        fixed_top
        + remaining["driver_code_norm"].astype(str).tolist()
    )


def build_future_feature_frame(
    warehouse,
    event_norm,
    race_round,
    grid_codes,
    target_year
):

    validate_warehouse(warehouse)

    grid = [
        str(code).upper().strip()
        for code in grid_codes
    ]

    if len(grid) != 22:
        raise ValueError(
            f"Expected 22 drivers, received {len(grid)}."
        )

    if len(grid) != len(set(grid)):
        raise ValueError(
            "Each driver can occupy only one grid position."
        )

    roster_features = get_roster_as_of(
        warehouse,
        target_year=int(target_year),
        target_round=race_round,
        driver_codes=grid
    )

    roster_features["starting_grid"] = np.arange(
        1, 23, dtype=float
    )

    roster_features["event_norm"] = str(event_norm)

    feature_frame = roster_features[
        FEATURE_COLUMNS
    ].copy()

    feature_frame["starting_grid"] = pd.to_numeric(
        feature_frame["starting_grid"],
        errors="raise"
    )

    feature_frame["driver_code_norm"] = (
        feature_frame["driver_code_norm"].astype(str).str.upper()
    )

    feature_frame["constructor"] = (
        feature_frame["constructor"].fillna("Unknown").astype(str)
    )

    feature_frame["event_norm"] = (
        feature_frame["event_norm"].astype(str)
    )

    if feature_frame.columns.tolist() != FEATURE_COLUMNS:
        raise AssertionError(
            "Feature schema does not match V2.8."
        )

    if len(feature_frame) != 22:
        raise AssertionError(
            "Prediction frame must contain 22 drivers."
        )

    return feature_frame


def predict_event(
    models,
    feature_frame,
    driver_codes
):

    """
    Run all three saved V2.8 models.

    race_win_probability is a normalised relative model score,
    not a calibrated real-world probability.
    """

    if feature_frame.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError(
            "Prediction input does not match the V2.8 schema."
        )

    if len(feature_frame) != len(driver_codes):
        raise ValueError(
            "Driver count does not match prediction input."
        )

    predictions = {}

    for name, model in models.items():

        raw = np.asarray(
            model.predict_proba(
                feature_frame
            )[:, 1],
            dtype=float
        )

        raw = np.clip(raw, 1e-8, 1 - 1e-8)

        logits = np.log(
            raw / (1 - raw)
        )

        exp_scores = np.exp(
            logits - np.max(logits)
        )

        relative_scores = (
            exp_scores / exp_scores.sum()
        )

        result = pd.DataFrame({
            "driver_code": [
                str(code).upper()
                for code in driver_codes
            ],
            "starting_grid": np.arange(
                1, len(driver_codes) + 1
            ),
            "raw_win_score": raw,
            "race_win_probability": relative_scores
        })

        result = result.sort_values(
            "race_win_probability",
            ascending=False,
            kind="mergesort"
        ).reset_index(drop=True)

        result["predicted_rank"] = np.arange(
            1, len(result) + 1
        )

        predictions[name] = result

    return predictions
