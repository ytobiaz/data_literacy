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
    """Extract start/end points from line segments.
    
    Returns GeoDataFrame with one row per endpoint, including segment ID and endpoint role (start/end).
    """

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
    """Cluster and snap endpoint points to grid of size tol_m.
    
    Returns NodeClustering with clustered node geometries, point centroids, and segment-node mapping.
    """

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
    """Filter nodes by degree (unique segments connected) to identify crossings.
    
    Returns Series of node_ids where degree >= min_degree (default: 3+ segments = intersection).
    """

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
    """Spatially join accidents to nearest crossing nodes and aggregate to node×year×month.
    
    Returns tuple: (accident-level data with node_id assignment, aggregated panel by node×year×month).
    """

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

    # Aggregate with rich factor distributions (like segment-level)
    acc_node_ym = _aggregate_accidents_node_year_month_rich(assigned)

    return acc_node, acc_node_ym


def _aggregate_accidents_node_year_month_rich(
    assigned: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate to node×year×month with total counts and categorical distributions (_rich version) - counts+shares."""
    
    acc = assigned.copy()
    
    if "_accident_row_id" not in acc.columns:
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
    
    keys = ["node_id", "year", "month"]
    
    # Categorical columns (incident types/factors)
    cat_cols = [
        "injury_severity",
        "accident_kind",
        "accident_type",
        "light_condition",
        "road_condition"
    ]
    cat_cols = [c for c in cat_cols if c in acc.columns]
    
    # Total accidents count
    acc_base = (
        acc.groupby(keys, observed=True)
        .agg(total_accidents=("_accident_row_id", "size"))
        .reset_index()
    )
    
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
    
    return acc_base.merge(acc_cats, on=keys, how="left")


def build_node_exposure_panel_from_segments(
    segment_exposure_ym: pd.DataFrame,
    segment_node_map: pd.DataFrame,
    crossing_ids: Iterable[int],
    *,
    trip_col: str = "sum_strava_total_trip_count",
    counter_col: str = "counter_name",
) -> pd.DataFrame:
    """Aggregate segment exposure to node×year×month by summing a trip_col.
    
    Applies 0.5 correction factor to account for double-counting:
    each segment contributes to 2 nodes (start/end).
    """

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
    
    # Account for double-counting: each segment contributes to 2 nodes (start/end)
    node_exposure_ym["monthly_strava_trips"] = node_exposure_ym["monthly_strava_trips"] * 0.5

    return node_exposure_ym


def print_node_exposure_quality_summary(
    node_exposure_ym: pd.DataFrame,
    crossings_gdf: GeoDataFrame,
    *,
    trip_col: str = "monthly_strava_trips",
) -> None:
    """Print quality summary for node exposure panel: coverage, temporal range, trip volume stats, and missing/zero checks."""
    
    quality_results = {}
    
    # 1. Dataset structure check
    total_records = len(node_exposure_ym)
    unique_crossings = node_exposure_ym['node_id'].nunique()
    total_crossings = len(crossings_gdf)
    
    quality_results['dataset_structure'] = {
        'status': 'PASS',
        'message': f'{total_records:,} records (crossing × year × month combinations)',
        'value': f'{unique_crossings:,} unique crossings'
    }
    
    # 2. Coverage check (crossings with ANY exposure data)
    # NOTE: This shows how many crossings have at least one record in the panel.
    # Missing crossings (100% - coverage%) have NO records at all (not present in dataset).
    coverage_pct = (unique_crossings / total_crossings * 100) if total_crossings > 0 else 0
    status = 'PASS' if coverage_pct >= 90 else 'WARN'
    missing_crossings = total_crossings - unique_crossings
    quality_results['crossing_coverage'] = {
        'status': status,
        'message': f'{unique_crossings:,} / {total_crossings:,} crossings present in panel ({missing_crossings:,} crossings absent)',
        'value': f'{coverage_pct:.1f}%'
    }
    
    # 3. Temporal coverage check
    if 'year' in node_exposure_ym.columns and 'month' in node_exposure_ym.columns:
        min_year = node_exposure_ym['year'].min()
        max_year = node_exposure_ym['year'].max()
        expected_months = int((max_year - min_year + 1) * 12)
        unique_ym = node_exposure_ym[['year', 'month']].drop_duplicates()
        actual_months = len(unique_ym)
        
        temporal_status = 'PASS' if actual_months == expected_months else 'WARN'
        quality_results['temporal_coverage'] = {
            'status': temporal_status,
            'message': f'Years: {min_year:.0f}–{max_year:.0f}, {actual_months} / {expected_months} expected year-month combinations',
            'value': f'{actual_months} months'
        }
    
    # 4. Trip volume statistics (per month per crossing)
    if trip_col in node_exposure_ym.columns:
        trip_stats = node_exposure_ym[trip_col].describe()
        quality_results['trip_volume_per_month'] = {
            'status': 'INFO',
            'message': f'Trips/month per crossing: Mean={trip_stats["mean"]:,.0f}, Median={trip_stats["50%"]:,.0f}, Std={trip_stats["std"]:,.0f}',
            'value': f'Range: {trip_stats["min"]:,.0f}–{trip_stats["max"]:,.0f}'
        }
        
        # 5. Missing values check (NaN in trip column among existing records)
        # NOTE: This checks for NaN values in existing records, NOT for absent crossings
        missing_trips = node_exposure_ym[trip_col].isna().sum()
        missing_pct = (missing_trips / len(node_exposure_ym) * 100) if len(node_exposure_ym) > 0 else 0
        missing_status = 'PASS' if missing_trips == 0 else 'WARN'
        quality_results['no_missing_values_in_records'] = {
            'status': missing_status,
            'message': f'{missing_trips:,} records with NaN trip values (among {total_records:,} existing records)',
            'value': f'{missing_pct:.2f}%'
        }
        
        # 6. Zero trip records check (per month)
        zero_trips_per_month = (node_exposure_ym[trip_col] == 0).sum()
        zero_pct_per_month = (zero_trips_per_month / len(node_exposure_ym) * 100) if len(node_exposure_ym) > 0 else 0
        zero_status_per_month = 'INFO' if zero_trips_per_month > 0 else 'PASS'
        quality_results['zero_trip_records_per_month'] = {
            'status': zero_status_per_month,
            'message': f'{zero_trips_per_month:,} month-records with zero trips (out of {total_records:,} total records)',
            'value': f'{zero_pct_per_month:.2f}%'
        }
        
        # 7. Zero trip crossings check (total across all time)
        crossing_totals = node_exposure_ym.groupby('node_id')[trip_col].sum()
        zero_crossings_total = (crossing_totals == 0).sum()
        zero_crossings_pct = (zero_crossings_total / unique_crossings * 100) if unique_crossings > 0 else 0
        zero_total_status = 'WARN' if zero_crossings_total > 0 else 'PASS'
        quality_results['zero_trip_crossings_all_time'] = {
            'status': zero_total_status,
            'message': f'{zero_crossings_total:,} crossings with zero total trips across all years (2019-2023)',
            'value': f'{zero_crossings_pct:.2f}%'
        }
    
    # Print results in same format as plot_merged_panel_quality_overview
    print("\n" + "="*20 + "NODE EXPOSURE QUALITY SUMMARY" + "="*20)
    for check_name, result in quality_results.items():
        print(f"\n[{result['status']}] {check_name}")
        print(f"   {result['message']}")
        print(f"   Value: {result['value']}")

