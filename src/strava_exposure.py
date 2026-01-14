from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import data_path

columns_to_keep = ['counter_name',
 'date',
 'count',
 'year',
 'latitude',
 'longitude',
 'geometry',
 'socioeconomic_gender_distribution',
 'infrastructure_bicyclelane_type',
 'infrastructure_type_of_street',
 'infrastructure_number_of_street_lanes',
 'infrastructure_street_smoothness',
 'infrastructure_street_surface',
 'infrastructure_max_speed',
 'infrastructure_cyclability',
 #'weather_temp_avg',
 #'weather_temp_min',
 #'weather_temp_max',
 #'weather_precipitation',
 #'weather_snowfall',
 #'weather_wind_speed_avg',
 #'weather_wind_speed_gust',
 #'weather_pressure',
 #'weather_sunshine_duration',
 #'strava_total_trip_count',
 #'strava_ride_count',
 #'day_of_week',
 #'month'
 ]

def load_and_aggregate_monthly_strava_counts_per_segment():
    """Load Strava Berlin data and aggregate counts to monthly level per segment."""
    parquet_path=data_path("strava", "berlin_data.parquet")
    # only load necessary columns to reduce memory usage
    df = pd.read_parquet(path=parquet_path, columns=['counter_name', 'date', 'count'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    agg_df = (
        df.groupby(['counter_name', 'year', 'month'], as_index=False)
        .agg({'count': 'sum'})
    )

    return agg_df



def load_strava_berlin_data(parquet_path: str | Path | None = None) -> pd.DataFrame:
    parquet_path = (
        Path(parquet_path) if parquet_path is not None else data_path("strava", "berlin_data.parquet")
    )
    return pd.read_parquet(parquet_path, columns=columns_to_keep)


def column_stability_summary(df: pd.DataFrame, *, group_col: str = "counter_name") -> pd.DataFrame:
    """For each column, compute whether it varies within a segment over time.

    Returns a dataframe with columns: column, segments_total, segments_varying, max_unique_within_any_segment.
    """

    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    grp = df.groupby(group_col, sort=False)

    summary: list[dict[str, object]] = []
    for col in df.columns:
        if col == group_col:
            continue
        nunique = grp[col].nunique(dropna=True)
        varying = nunique.gt(1)
        summary.append(
            {
                "column": col,
                "segments_total": int(len(nunique)),
                "segments_varying": int(varying.sum()),
                "max_unique_within_any_segment": int(nunique.max()) if len(nunique) else 0,
            }
        )

    return (
        pd.DataFrame(summary)
        .sort_values("segments_varying", ascending=True)
        .reset_index(drop=True)
    )


def build_exposure_panel_segment_year_month(
    strava_berlin_data: pd.DataFrame,
    *,
    segment_static: pd.DataFrame,
    summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate Strava data to segment–year–month and attach canonical geometry."""

    # Avoid copying the full (very large) dataframe; only materialize the columns we need.
    df = strava_berlin_data

    required = ["counter_name", "date", "latitude", "longitude", "street_name", "is_shortterm"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        raise KeyError(f"Strava dataframe missing required columns: {missing_req}")

    df = df.dropna(subset=["latitude", "longitude", "street_name", "is_shortterm"])

    # Compute time keys without storing an extra 'date' copy.
    date = pd.to_datetime(df["date"], errors="coerce")
    year = date.dt.year
    month = date.dt.month

    keys = ["counter_name", "year", "month"]

    if summary_df is None:
        summary_df = column_stability_summary(df, group_col="counter_name")

    constant_cols_raw = summary_df.loc[summary_df["segments_varying"] == 0, "column"].tolist()
    constant_cols = [c for c in constant_cols_raw if c in df.columns and c != "geometry"]

    sum_cols = [
        c
        for c in [
            "count",
            "strava_total_trip_count",
            "strava_ride_count",
            "strava_ebike_ride_count",
            "strava_total_people_count",
            "strava_total_commute_trip_count",
            "strava_total_leisure_trip_count",
            "strava_total_morning_trip_count",
            "strava_total_midday_trip_count",
            "strava_total_evening_trip_count",
            "strava_total_overnight_trip_count",
            "strava_total_male_people_count",
            "strava_total_female_people_count",
            "strava_total_18_34_people_count",
            "strava_total_35_54_people_count",
            "strava_total_55_64_people_count",
            "strava_total_65_plus_people_count",
            "strava_total_unspecified_people_count",
            "motorized_vehicle_count_all_vehicles_6km",
            "motorized_vehicle_count_cars_6km",
            "motorized_vehicle_count_trucks_6km",
            "motorized_vehicle_count_all_vehicles",
            "motorized_vehicle_count_cars",
            "motorized_vehicle_count_trucks",
        ]
        if c in df.columns
    ]

    mean_cols = [
        c
        for c in [
            "strava_total_average_speed_meters_per_second",
            "motorized_avg_speed_all_vehicles_6km",
            "motorized_avg_speed_cars_6km",
            "motorized_avg_speed_trucks_6km",
            "motorized_avg_speed_all_vehicles",
            "motorized_avg_speed_cars",
            "motorized_avg_speed_trucks",
            "infrastructure_distance_citycenter_km",
        ]
        + [c for c in df.columns if c.startswith("weather_")]
        + [c for c in df.columns if c.startswith("socioeconomic_")]
        if c in df.columns
    ]

    cat_cols = [c for c in ["strava_activity_type"] if c in df.columns]

    # Build the varying frame explicitly to keep peak memory low.
    vary_cols = ["counter_name"] + sum_cols + mean_cols + cat_cols
    df_var = df[vary_cols].copy()
    df_var["year"] = year.to_numpy()
    df_var["month"] = month.to_numpy()

    df_const = (
        df[["counter_name"] + constant_cols]
        .drop_duplicates("counter_name")
        .reset_index(drop=True)
    )

    for k in keys:
        df_var[k] = df_var[k].astype("category")

    def fast_mode(s: pd.Series):
        vc = s.value_counts(dropna=True)
        return vc.index[0] if not vc.empty else pd.NA

    agg_map = {
        **{c: "sum" for c in sum_cols},
        **{c: "mean" for c in mean_cols},
        **{c: fast_mode for c in cat_cols},
    }

    agg_segment_ym = df_var.groupby(keys, sort=False, observed=True).agg(agg_map).reset_index()

    rename_map: dict[str, str] = {}
    rename_map.update({c: f"sum_{c}" for c in sum_cols if c in agg_segment_ym.columns})
    rename_map.update({c: f"mean_{c}" for c in mean_cols if c in agg_segment_ym.columns})
    rename_map.update({c: f"mode_{c}" for c in cat_cols if c in agg_segment_ym.columns})

    agg_segment_ym = agg_segment_ym.rename(columns=rename_map)

    final_exposure_ym = agg_segment_ym.merge(df_const, on="counter_name", how="left")

    if "geometry" in segment_static.columns:
        final_exposure_ym = final_exposure_ym.merge(
            segment_static[["counter_name", "geometry"]],
            on="counter_name",
            how="left",
        )

    final_exposure_ym["year"] = final_exposure_ym["year"].astype("int64")
    final_exposure_ym["month"] = final_exposure_ym["month"].astype("int64")

    return final_exposure_ym
