from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import geopandas as gpd
from geopandas import GeoDataFrame

from .utils import data_path

def assign_accidents_to_nearest_segment(
    accidents_bike_berlin: pd.DataFrame,
    segments_gdf: GeoDataFrame,
    *,
    canonical_crs: str = "EPSG:32633",
    max_distance_m: float = 10.0,
    x_col: str = "XGCSWGS84",
    y_col: str = "YGCSWGS84",
) -> GeoDataFrame:
    """Assign each accident to exactly one nearest segment (within max_distance_m)."""

    accidents = accidents_bike_berlin.reset_index(drop=True).copy()

    if x_col not in accidents.columns or y_col not in accidents.columns:
        raise KeyError(f"Expected coordinate columns {x_col!r}, {y_col!r} in accidents dataframe")

    accident_locations_gdf = gpd.GeoDataFrame(
        accidents,
        geometry=gpd.points_from_xy(accidents[x_col], accidents[y_col]),
        crs="EPSG:4326",
    ).to_crs(canonical_crs)

    accident_locations_gdf = accident_locations_gdf.reset_index(drop=True)
    accident_locations_gdf["acc_id"] = accident_locations_gdf.index

    joined = gpd.sjoin_nearest(
        accident_locations_gdf,
        segments_gdf,
        how="left",
        max_distance=max_distance_m,
        distance_col="dist",
    )

    joined = joined.dropna(subset=["index_right"]).copy()

    joined_nearest_unique = (
        joined.sort_values("dist").drop_duplicates(subset=["acc_id"], keep="first")
    )

    return joined_nearest_unique
