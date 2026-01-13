from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .accidents import ACCIDENT_COLUMNS_EN


def aggregate_accidents_segment_year_month_rich(
    joined_nearest_unique: pd.DataFrame,
    *,
    column_map: Mapping[str, str] = ACCIDENT_COLUMNS_EN,
    exposure_year_min: int | None = None,
    exposure_year_max: int | None = None,
) -> pd.DataFrame:
    """Aggregate accidents to segment–year–month with rich distributions (counts+shares)."""

    acc = joined_nearest_unique.copy()
    acc = acc.rename(columns=dict(column_map))

    for col in ["year", "month"]:
        if col in acc.columns:
            acc[col] = acc[col].astype("int64")

    keys = ["counter_name", "year", "month"]

    if exposure_year_min is not None and "year" in acc.columns:
        acc = acc[acc["year"].ge(exposure_year_min)].copy()
    if exposure_year_max is not None and "year" in acc.columns:
        acc = acc[acc["year"].le(exposure_year_max)].copy()

    # Ensure we have a non-null identifier for counting/pivoting.
    # Some source CSVs appear to have missing/blank UIDENTSTLA, so accident_id can be all-NA.
    id_candidates = ["accident_id", "accident_id_extended", "acc_id"]
    id_col = next(
        (
            c
            for c in id_candidates
            if c in acc.columns and not acc[c].isna().all()
        ),
        None,
    )
    if id_col is None:
        acc["_accident_row_id"] = np.arange(len(acc), dtype="int64")
    else:
        acc["_accident_row_id"] = acc[id_col]
        if acc["_accident_row_id"].isna().any():
            acc.loc[acc["_accident_row_id"].isna(), "_accident_row_id"] = (
                np.arange(len(acc), dtype="int64")[acc["_accident_row_id"].isna().to_numpy()]
            )

    flag_cols = [
        "involved_bicycle",
        "involved_passenger_car",
        "involved_pedestrian",
        "involved_motorcycle",
        "involved_goods_vehicle",
        "involved_other_vehicle",
        "involved_road",
        "road_condition_flag",
    ]
    flag_cols = [c for c in flag_cols if c in acc.columns]

    cat_cols = [
        "injury_severity",
        "accident_kind",
        "accident_type",
        "light_condition",
        "road_condition",
    ]
    cat_cols = [c for c in cat_cols if c in acc.columns]

    acc_base = (
        acc.groupby(keys, observed=True)
        .agg(total_accidents=("_accident_row_id", "size"))
        .reset_index()
    )

    # Flags: counts + shares
    if flag_cols:
        acc_flags_counts = acc.groupby(keys, observed=True)[flag_cols].sum().reset_index()
        acc_flags_counts = acc_flags_counts.rename(columns={c: f"acc_{c}_count" for c in flag_cols})

        acc_flags = acc_base[keys + ["total_accidents"]].merge(acc_flags_counts, on=keys, how="left")

        for c in flag_cols:
            cnt_col = f"acc_{c}_count"
            share_col = f"acc_{c}_share"
            acc_flags[share_col] = acc_flags[cnt_col] / acc_flags["total_accidents"].replace(0, pd.NA)

        acc_flags = acc_flags.drop(columns=["total_accidents"])
    else:
        acc_flags = acc_base[keys].copy()

    # Categoricals: pivot counts + shares
    cat_blocks: list[pd.DataFrame] = []
    for col in cat_cols:
        pivot_counts = acc.pivot_table(
            index=keys,
            columns=col,
            values="_accident_row_id",
            aggfunc="count",
            fill_value=0,
        )
        pivot_counts.columns = [f"acc_{col}_count_{str(cat)}" for cat in pivot_counts.columns]

        row_sums = pivot_counts.sum(axis=1).replace(0, pd.NA)
        pivot_shares = pivot_counts.div(row_sums, axis=0)
        pivot_shares.columns = [name.replace("_count_", "_share_") for name in pivot_counts.columns]

        cat_blocks.append(pd.concat([pivot_counts, pivot_shares], axis=1).reset_index())

    if cat_blocks:
        from functools import reduce

        acc_cats = reduce(lambda left, right: left.merge(right, on=keys, how="outer"), cat_blocks)
    else:
        acc_cats = acc_base[keys].copy()

    return acc_base.merge(acc_flags, on=keys, how="left").merge(acc_cats, on=keys, how="left")


def merge_exposure_and_accidents(
    final_exposure_ym: pd.DataFrame,
    accidents_agg_ym_rich: pd.DataFrame,
    *,
    merge_keys: Sequence[str] = ("counter_name", "year", "month"),
) -> pd.DataFrame:
    merged = final_exposure_ym.merge(
        accidents_agg_ym_rich,
        on=list(merge_keys),
        how="left",
        validate="one_to_one",
    )

    acc_cols = [c for c in accidents_agg_ym_rich.columns if c not in merge_keys]
    merged[acc_cols] = merged[acc_cols].fillna(0)
    return merged


def build_core_risk_panel(
    merged_accidents_strava_ym: pd.DataFrame,
    *,
    trip_col: str = "sum_strava_total_trip_count",
    sensor_col: str = "sum_count",
) -> pd.DataFrame:
    full = merged_accidents_strava_ym.copy()

    key_cols = ["counter_name", "year", "month"]
    segment_cols = [c for c in ["geometry"] if c in full.columns]
    exposure_cols = [c for c in [trip_col, sensor_col] if c in full.columns]
    acc_core_cols = [c for c in ["total_accidents"] if c in full.columns]

    severity_cols = [
        c
        for c in full.columns
        if c.startswith("acc_injury_severity_count_") or c.startswith("acc_injury_severity_share_")
    ]

    accident_type_cols = [
        c
        for c in full.columns
        if c.startswith("acc_accident_type_count_") or c.startswith("acc_accident_type_share_")
    ]

    accident_kind_cols = [
        c
        for c in full.columns
        if c.startswith("acc_accident_kind_count_") or c.startswith("acc_accident_kind_share_")
    ]

    light_cols = [
        c
        for c in full.columns
        if c.startswith("acc_light_condition_count_") or c.startswith("acc_light_condition_share_")
    ]

    road_cols = [
        c
        for c in full.columns
        if c.startswith("acc_road_condition_count_") or c.startswith("acc_road_condition_share_")
    ]

    cols_keep = (
        key_cols
        + segment_cols
        + exposure_cols
        + acc_core_cols
        + severity_cols
        + accident_type_cols
        + accident_kind_cols
        + light_cols
        + road_cols
    )

    seen: set[str] = set()
    cols_keep = [c for c in cols_keep if not (c in seen or seen.add(c))]

    core_panel = full[cols_keep].copy()

    if trip_col in core_panel.columns:
        core_panel["monthly_strava_trips"] = core_panel[trip_col]

    if "total_accidents" in core_panel.columns and "monthly_strava_trips" in core_panel.columns:
        denom = core_panel["monthly_strava_trips"].replace(0, np.nan)
        core_panel["risk_accidents_per_trip"] = core_panel["total_accidents"] / denom
        core_panel["risk_accidents_per_10k_trips"] = core_panel["risk_accidents_per_trip"] * 10_000

    return core_panel


def sanity_check_merge(
    *,
    merged_accidents_strava_ym: pd.DataFrame,
    accidents_agg_ym_rich: pd.DataFrame,
    final_exposure_ym: pd.DataFrame,
    merge_keys: Sequence[str] = ("counter_name", "year", "month"),
) -> dict[str, object]:
    merge_keys = list(merge_keys)

    exposure_duplicates = int(final_exposure_ym.duplicated(subset=merge_keys).sum())
    if exposure_duplicates:
        raise AssertionError(f"Found {exposure_duplicates} duplicate keys in Strava exposure table")

    exposure_index = pd.MultiIndex.from_frame(final_exposure_ym[merge_keys]).unique()
    accident_index = pd.MultiIndex.from_frame(accidents_agg_ym_rich[merge_keys]).unique()

    segments_with_accidents = int(exposure_index.isin(accident_index).sum())
    segments_without_accidents = int(len(exposure_index) - segments_with_accidents)

    accidents_missing_mask = ~accident_index.isin(exposure_index)
    missing_count = int(accidents_missing_mask.sum())

    merged_total = (
        float(merged_accidents_strava_ym["total_accidents"].sum())
        if "total_accidents" in merged_accidents_strava_ym.columns
        else float("nan")
    )
    source_total = (
        float(accidents_agg_ym_rich["total_accidents"].sum())
        if "total_accidents" in accidents_agg_ym_rich.columns
        else float("nan")
    )

    return {
        "segments_with_accidents": segments_with_accidents,
        "segments_without_accidents": segments_without_accidents,
        "accident_groups_missing_exposure": missing_count,
        "merged_total_accidents": merged_total,
        "source_total_accidents": source_total,
        "lost_accidents_due_to_missing_exposure": source_total - merged_total if np.isfinite(source_total) and np.isfinite(merged_total) else None,
    }
