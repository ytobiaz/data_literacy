from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import geopandas as gpd
from geopandas import GeoDataFrame
import shapely.wkt

from .utils import data_path


@dataclass(frozen=True)
class SegmentGeometry:
    segments_gdf: GeoDataFrame
    segment_static: GeoDataFrame
    canonical_crs: str


def load_segment_geometry(
    parquet_path: str | Path | None = None,
    *,
    canonical_crs: str = "EPSG:32633",
    source_crs: str = "EPSG:4326",
) -> SegmentGeometry:
    """Load canonical Strava segment geometries and reproject to a metric CRS."""

    parquet_path = (
        Path(parquet_path)
        if parquet_path is not None
        else data_path("strava", "berlin_graph_geometry.parquet")
    )

    segment_geo_df = pd.read_parquet(parquet_path).copy()
    if "geometry" not in segment_geo_df.columns:
        raise KeyError("Expected a 'geometry' column in the segment geometry parquet")

    segment_geo_df["geometry"] = segment_geo_df["geometry"].apply(shapely.wkt.loads)

    segment_geo_gdf = gpd.GeoDataFrame(
        segment_geo_df,
        geometry="geometry",
        crs=source_crs,
    ).to_crs(canonical_crs)

    cols_static = ["counter_name", "geometry"]
    if "street_name" in segment_geo_gdf.columns:
        cols_static.append("street_name")

    segment_static = (
        segment_geo_gdf[cols_static]
        .drop_duplicates("counter_name")
        .reset_index(drop=True)
    )

    return SegmentGeometry(
        segments_gdf=segment_geo_gdf,
        segment_static=segment_static,
        canonical_crs=canonical_crs,
    )
