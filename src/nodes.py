from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point


@dataclass(frozen=True)
class NodeClustering:
    nodes_raw: GeoDataFrame
    node_points: GeoDataFrame
    segment_node_map: pd.DataFrame
    tol_m: float


def build_nodes_from_segment_endpoints(
    segments_gdf: GeoDataFrame,
    *,
    counter_col: str = "counter_name",
) -> GeoDataFrame:
    """Create a point-per-endpoint table from segment geometries."""

    if counter_col not in segments_gdf.columns:
        raise KeyError(f"Missing {counter_col!r} in segments_gdf")
    if "geometry" not in segments_gdf.columns:
        raise KeyError("Missing 'geometry' in segments_gdf")

    def _get_endpoints(geom):
        if geom is None or getattr(geom, "is_empty", False):
            return None, None
        geom_type = getattr(geom, "geom_type", None)
        if geom_type == "LineString":
            coords = list(geom.coords)
        elif geom_type == "MultiLineString":
            lines = list(geom.geoms)
            if not lines:
                return None, None
            longest = max(lines, key=lambda g: g.length)
            coords = list(longest.coords)
        else:
            return None, None

        if not coords:
            return None, None
        return coords[0], coords[-1]

    records: list[dict[str, object]] = []
    for row in segments_gdf[[counter_col, "geometry"]].itertuples(index=False):
        counter_name = getattr(row, counter_col)
        geom = getattr(row, "geometry")
        start_pt, end_pt = _get_endpoints(geom)
        if start_pt is not None:
            records.append({counter_col: counter_name, "role": "start", "geometry": Point(start_pt)})
        if end_pt is not None:
            records.append({counter_col: counter_name, "role": "end", "geometry": Point(end_pt)})

    return gpd.GeoDataFrame(records, geometry="geometry", crs=segments_gdf.crs)


def cluster_nodes_snap_grid(
    nodes_raw: GeoDataFrame,
    *,
    tol_m: float = 2.0,
    counter_col: str = "counter_name",
) -> NodeClustering:
    """Cluster endpoint points by snapping to a grid of size tol_m."""

    if "geometry" not in nodes_raw.columns:
        raise KeyError("Missing 'geometry' in nodes_raw")
    if counter_col not in nodes_raw.columns:
        raise KeyError(f"Missing {counter_col!r} in nodes_raw")
    if tol_m <= 0:
        raise ValueError("tol_m must be > 0")

    nodes = nodes_raw.copy()
    nodes["x_rounded"] = (nodes.geometry.x / tol_m).round().astype("int64")
    nodes["y_rounded"] = (nodes.geometry.y / tol_m).round().astype("int64")
    nodes["node_id"] = nodes.groupby(["x_rounded", "y_rounded"], sort=False).ngroup().astype("int64")

    node_points = nodes.dissolve(by="node_id", as_index=False)[["node_id", "geometry"]]

    # dissolve over points typically produces MultiPoint -> collapse to a single Point per node
    node_points["geometry"] = node_points.geometry.centroid

    segment_node_map = nodes[[counter_col, "node_id", "role"]].drop_duplicates()

    return NodeClustering(
        nodes_raw=nodes,
        node_points=node_points,
        segment_node_map=segment_node_map,
        tol_m=float(tol_m),
    )


def select_crossings_by_degree(
    nodes_raw: pd.DataFrame,
    *,
    min_degree: int = 3,
    counter_col: str = "counter_name",
) -> pd.Series:
    """Return node_ids that have at least min_degree unique segments touching them."""

    if "node_id" not in nodes_raw.columns:
        raise KeyError("Missing 'node_id' in nodes_raw")
    if counter_col not in nodes_raw.columns:
        raise KeyError(f"Missing {counter_col!r} in nodes_raw")
    if min_degree < 1:
        raise ValueError("min_degree must be >= 1")

    deg = nodes_raw.groupby("node_id", sort=False)[counter_col].nunique()
    return deg[deg >= min_degree].index.to_series(index=None)


def assign_accidents_to_nearest_crossing(
    accidents: pd.DataFrame,
    crossings_gdf: GeoDataFrame,
    *,
    max_distance_m: float = 20.0,
    geometry_col: str = "geometry",
) -> tuple[GeoDataFrame, pd.DataFrame]:
    """Assign accidents to nearest crossing node and aggregate to node×year×month."""

    for col in ["year", "month"]:
        if col not in accidents.columns:
            raise KeyError(f"Missing {col!r} in accidents")
    if geometry_col not in accidents.columns:
        raise KeyError(f"Missing {geometry_col!r} in accidents")
    if "node_id" not in crossings_gdf.columns:
        raise KeyError("Missing 'node_id' in crossings_gdf")

    acc_gdf = gpd.GeoDataFrame(accidents.copy(), geometry=geometry_col, crs=crossings_gdf.crs)

    # Avoid GeoPandas join column name collisions
    for df_ in (acc_gdf, crossings_gdf):
        df_.drop(columns=["index_right", "index_left"], errors="ignore", inplace=True)

    acc_gdf = acc_gdf.reset_index(drop=True)
    crossings_clean = crossings_gdf[["node_id", "geometry"]].reset_index(drop=True)

    acc_node = gpd.sjoin_nearest(
        acc_gdf,
        crossings_clean,
        how="left",
        max_distance=max_distance_m,
        distance_col="dist_node",
        rsuffix="node",
    )

    # Keep ALL accidents (within your study area / segment corridor),
    # and mark whether a crossing was found within max_distance_m.
    acc_node["has_crossing"] = acc_node["node_id"].notna()

    # Aggregate ONLY those accidents that actually got assigned to a crossing
    assigned = acc_node.dropna(subset=["node_id"]).copy()
    assigned["node_id"] = assigned["node_id"].astype("int64")

    acc_node_ym = (
        assigned.groupby(["node_id", "year", "month"], observed=True)
        .size()
        .reset_index(name="total_accidents")
    )

    return acc_node, acc_node_ym


def build_node_exposure_panel_from_segments(
    segment_exposure_ym: pd.DataFrame,
    segment_node_map: pd.DataFrame,
    crossing_ids: Iterable[int],
    *,
    trip_col: str = "sum_strava_total_trip_count",
    counter_col: str = "counter_name",
) -> pd.DataFrame:
    """Aggregate segment exposure to node×year×month by summing a trip_col."""

    for col in [counter_col, "year", "month", trip_col]:
        if col not in segment_exposure_ym.columns:
            raise KeyError(f"Missing {col!r} in segment_exposure_ym")
    for col in [counter_col, "node_id"]:
        if col not in segment_node_map.columns:
            raise KeyError(f"Missing {col!r} in segment_node_map")

    crossing_ids = set(int(x) for x in crossing_ids)

    segment_node_map_cross = segment_node_map[segment_node_map["node_id"].isin(crossing_ids)].copy()

    # Prevent double-counting when a segment maps to the same node more than once (e.g., start/end collapse)
    segment_node_map_cross = segment_node_map_cross.drop_duplicates(subset=[counter_col, "node_id"])

    segment_exposure_nodes = segment_exposure_ym.merge(
        segment_node_map_cross[[counter_col, "node_id"]],
        on=counter_col,
        how="inner",
    )

    node_exposure_ym = (
        segment_exposure_nodes.groupby(["node_id", "year", "month"], observed=True)
        .agg(monthly_strava_trips=(trip_col, "sum"))
        .reset_index()
    )

    return node_exposure_ym


def build_node_risk_panel(
    node_exposure_ym: pd.DataFrame,
    acc_node_ym: pd.DataFrame,
    crossings_gdf: GeoDataFrame,
) -> GeoDataFrame:
    """Combine node exposure and node accidents into a node-level risk panel.

    IMPORTANT:
    We must keep the union of node×year×month from exposure and accidents.
    Otherwise accidents that occur in months without exposure rows disappear.
    """

    node_panel_ym = node_exposure_ym.merge(
        acc_node_ym,
        on=["node_id", "year", "month"],
        how="outer",
    )

    node_panel_ym["monthly_strava_trips"] = node_panel_ym["monthly_strava_trips"].fillna(0)
    node_panel_ym["total_accidents"] = node_panel_ym["total_accidents"].fillna(0)

    # Fill missing metrics
    if "monthly_strava_trips" in node_panel_ym.columns:
        node_panel_ym["monthly_strava_trips"] = node_panel_ym["monthly_strava_trips"].fillna(0)
    else:
        # If exposure panel uses a different column name, fail loudly rather than silently miscompute
        raise KeyError("Expected 'monthly_strava_trips' in node_exposure_ym")

    node_panel_ym["total_accidents"] = node_panel_ym["total_accidents"].fillna(0)

    # Add geometry
    node_panel_ym = node_panel_ym.merge(
        crossings_gdf[["node_id", "geometry"]],
        on="node_id",
        how="left",
    )

    # Risk is undefined when trips are 0 -> use NaN denominator
    denom = node_panel_ym["monthly_strava_trips"].replace(0, np.nan)
    node_panel_ym["risk_accidents_per_trip"] = node_panel_ym["total_accidents"] / denom
    node_panel_ym["risk_accidents_per_10k_trips"] = node_panel_ym["risk_accidents_per_trip"] * 10_000

    return gpd.GeoDataFrame(node_panel_ym, geometry="geometry", crs=crossings_gdf.crs)
