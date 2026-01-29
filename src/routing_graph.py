"""
Graph construction and risk attribute computation for routing.

This module provides utilities for building routing graphs from segment and junction data,
attaching risk attributes (segment risk and junction penalties), and preparing graphs
for shortest-path and constrained min-risk routing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString


# Type aliases
NodeKey = Tuple[float, float]


# Helper functions

def _edge_endpoints_as_node_keys(linestring: LineString, ndigits: int = 0) -> Tuple[NodeKey, NodeKey]:
    """
    Create stable node keys from LineString endpoints in a metric CRS.

    For UTM meters:
      ndigits=0 => 1m rounding (robust)
      ndigits=1 => 0.1m rounding (less robust)
      ndigits=3 => 1mm rounding (usually too strict)
    """
    (x1, y1) = linestring.coords[0]
    (x2, y2) = linestring.coords[-1]
    a = (round(float(x1), ndigits), round(float(y1), ndigits))
    b = (round(float(x2), ndigits), round(float(y2), ndigits))
    return a, b


def _risk_fallback_value(series: pd.Series, *, strategy: str = "median", default: float = 0.0) -> float:
    """
    Decide a safe fallback risk if a segment/node risk is NaN.

    strategy:
      - "median": median of observed values
      - "mean": mean of observed values
      - "zero": always 0
      - "high": conservative high value (90th percentile if possible)
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float(default)

    if strategy == "median":
        return float(np.nanmedian(s))
    if strategy == "mean":
        return float(np.nanmean(s))
    if strategy == "high":
        return float(np.nanpercentile(s, 90))
    if strategy == "zero":
        return 0.0

    return float(np.nanmedian(s))


def _build_node_risk_lookup(
    junction_panel_gdf: gpd.GeoDataFrame,
    *,
    node_id_col: str = "node_id",
    node_risk_col: str = "risk_weight_secure",
    fallback: float = 0.0,
    fallback_strategy: str = "median",
) -> Dict[str, float]:
    """
    Returns dict[node_id(str)] -> node_risk for pooled routing.

    Notes:
      - If junction_panel_gdf is empty: returns {} (caller chooses behavior).
      - If multiple rows per node_id exist (e.g., panel data): we aggregate by mean.
      - If a node risk value is NaN: filled by a fallback computed from observed values.
    """
    if junction_panel_gdf is None or len(junction_panel_gdf) == 0:
        return {}

    j = junction_panel_gdf.copy()
    if j.empty:
        return {}

    if node_id_col not in j.columns:
        raise KeyError(f"Missing node id column '{node_id_col}' in junction_panel_gdf")
    if node_risk_col not in j.columns:
        raise KeyError(f"Missing node risk column '{node_risk_col}' in junction_panel_gdf")

    # One row per node_id
    j = j.groupby(node_id_col, as_index=False)[node_risk_col].mean()

    fb = _risk_fallback_value(j[node_risk_col], strategy=fallback_strategy, default=fallback)
    j[node_risk_col] = pd.to_numeric(j[node_risk_col], errors="coerce").fillna(fb)

    return dict(zip(j[node_id_col].astype(str), j[node_risk_col].astype(float)))


def _attach_node_ids_to_graph_nodes(
    G: nx.Graph,
    crossings_gdf: gpd.GeoDataFrame,
    *,
    metric_epsg: int = 32633,
    node_id_col: str = "node_id",
    max_snap_m: float = 20.0,
) -> None:
    """
    Map your existing 'node_id' (junctions/crossings) onto graph nodes.

    We snap each crossing point to the nearest graph node (within max_snap_m).
    Stores: G.nodes[node_key]["node_id"] = <id>

    This is required for applying junction risk penalties.
    """
    if crossings_gdf is None or len(crossings_gdf) == 0:
        return

    if node_id_col not in crossings_gdf.columns:
        raise KeyError(f"Missing '{node_id_col}' in crossings_gdf")

    crossings_m = crossings_gdf.to_crs(epsg=metric_epsg).copy()

    graph_nodes = list(G.nodes())
    if len(graph_nodes) == 0:
        return

    graph_xy = np.array(graph_nodes, dtype=float)

    for _, row in crossings_m.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue

        nid = str(row[node_id_col])
        x, y = float(row.geometry.x), float(row.geometry.y)

        d2 = (graph_xy[:, 0] - x) ** 2 + (graph_xy[:, 1] - y) ** 2
        j = int(d2.argmin())
        dist = float(math.sqrt(d2[j]))
        if dist <= max_snap_m:
            node_key = graph_nodes[j]
            G.nodes[node_key]["node_id"] = nid


def _iter_edges(G: nx.Graph):
    """
    Unified iterator over edges returning (u, v, key, data) where key is None for non-multigraphs.
    """
    if isinstance(G, nx.MultiGraph):
        for u, v, k, data in G.edges(keys=True, data=True):
            yield u, v, k, data
    else:
        for u, v, data in G.edges(data=True):
            yield u, v, None, data


def _get_edge_data_for_step(G: nx.Graph, a, b, *, choose_by: str = "length_m") -> dict:
    """
    For MultiGraph: pick the parallel edge (between a,b) that minimizes choose_by
    (fallback to length_m).
    For Graph: return edge data.
    """
    if isinstance(G, nx.MultiGraph):
        edges = G.get_edge_data(a, b)
        if edges is None:
            raise KeyError(f"No edge between {a} and {b}")

        best = None
        best_val = None
        for _, d in edges.items():
            val = d.get(choose_by, d.get("length_m", 0.0))
            if best is None or val < best_val:
                best = d
                best_val = val
        return best
    else:
        d = G.get_edge_data(a, b)
        if d is None:
            raise KeyError(f"No edge between {a} and {b}")
        return d


# Configuration dataclasses

@dataclass(frozen=True)
class GraphBuildConfig:
    """Configuration for graph building from segment data."""
    metric_epsg: int = 32633
    endpoint_ndigits: int = 0
    seg_id_col: str = "counter_name"
    seg_exposure_col_candidates: Tuple[str, ...] = ("monthly_strava_trips", "sum_strava_total_trip_count")
    seg_risk_col_candidates: Tuple[str, ...] = ("risk_weight_secure",)
    risk_fallback_strategy: str = "median"   # median/mean/high/zero
    risk_fallback_default: float = 0.0
    drop_edges_with_zero_exposure: bool = False  # pooled setup typically disables this
    keep_parallel_edges: bool = True


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for risk computation on graph edges."""
    eta: float = 1.0  # weight of junction penalty inside the R(P) objective


@dataclass(frozen=True)
class RoutingGraphArtifacts:
    """Output from building a routing graph."""
    G: nx.Graph
    segment_risk_col_used: str
    node_risk_col_used: Optional[str]
    notes: str


# Graph construction

def build_routing_graph(
    segments_panel_gdf: gpd.GeoDataFrame,
    *,
    config: GraphBuildConfig = GraphBuildConfig(),
) -> nx.Graph:
    """
    Build a pooled routing graph from segment LineStrings.

    Nodes: segment endpoints (rounded in metric CRS).
    Edges carry:
      - segment_id
      - length_m
      - seg_risk
      - geometry

    If segments_panel_gdf contains multiple rows per segment (e.g., a panel),
    this function aggregates to one row per segment_id.
    """
    df0 = segments_panel_gdf
    if df0 is None or len(df0) == 0:
        raise ValueError("No segment rows found in segments_panel_gdf.")
    if df0.geometry is None:
        raise ValueError("segments_panel_gdf must have a geometry column.")

    # Pick risk column from the raw input
    risk_col = None
    for c in config.seg_risk_col_candidates:
        if c in df0.columns:
            risk_col = c
            break
    if risk_col is None:
        raise KeyError(f"Missing risk column. Tried: {config.seg_risk_col_candidates}")

    # Exposure column is optional in pooled mode
    exposure_col = None
    for c in config.seg_exposure_col_candidates:
        if c in df0.columns:
            exposure_col = c
            break

    # Aggregate to one row per segment_id
    agg = {"geometry": ("geometry", "first"), risk_col: (risk_col, "mean")}
    if exposure_col is not None:
        agg[exposure_col] = (exposure_col, "sum")

    df = df0.groupby(config.seg_id_col, as_index=False).agg(**agg)

    # Restore GeoDataFrame and project to metric CRS
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=segments_panel_gdf.crs)
    df_m = df.to_crs(epsg=config.metric_epsg).copy()

    # Optional exposure filtering (only if exposure exists)
    if config.drop_edges_with_zero_exposure and exposure_col is not None:
        df_m[exposure_col] = pd.to_numeric(df_m[exposure_col], errors="coerce")
        df_m = df_m[df_m[exposure_col].fillna(0) > 0].copy()

    # Fallback risk computed over the pooled panel
    fb = _risk_fallback_value(df_m[risk_col], strategy=config.risk_fallback_strategy, default=config.risk_fallback_default)

    # Build graph
    G = nx.MultiGraph() if config.keep_parallel_edges else nx.Graph()

    for _, row in df_m.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if not isinstance(geom, LineString):
            # If MultiLineString, take the longest component
            try:
                parts = list(geom.geoms)
                parts = [p for p in parts if isinstance(p, LineString) and not p.is_empty]
                if not parts:
                    continue
                geom = max(parts, key=lambda g: g.length)
            except Exception:
                continue

        u, v = _edge_endpoints_as_node_keys(geom, ndigits=config.endpoint_ndigits)
        seg_id = str(row[config.seg_id_col])
        length_m = float(geom.length)

        seg_risk = pd.to_numeric(row[risk_col], errors="coerce")
        if pd.isna(seg_risk):
            seg_risk = fb
        seg_risk = float(seg_risk)

        if u not in G:
            G.add_node(u, x=u[0], y=u[1])
        if v not in G:
            G.add_node(v, x=v[0], y=v[1])

        G.add_edge(
            u,
            v,
            segment_id=seg_id,
            length_m=length_m,
            seg_risk=seg_risk,
            geometry=geom,
        )

    if G.number_of_edges() == 0:
        raise ValueError("Graph has 0 edges after filtering. Check geometry and risk inputs.")

    return G


# Risk attribute computation

def add_node_penalty_attributes(
    G: nx.Graph,
    *,
    node_risk_by_node_id: Optional[Dict[str, float]] = None,
) -> None:
    """
    Add node_penalty attribute to each edge based on junction risks at endpoints.

    node_penalty = 0.5 * (risk_u + risk_v)
    """
    if node_risk_by_node_id is None:
        node_risk_by_node_id = {}

    for u, v, k, data in _iter_edges(G):
        uid = G.nodes[u].get("node_id")
        vid = G.nodes[v].get("node_id")
        ru = float(node_risk_by_node_id.get(uid, 0.0)) if uid is not None else 0.0
        rv = float(node_risk_by_node_id.get(vid, 0.0)) if vid is not None else 0.0
        node_penalty = 0.5 * (ru + rv)

        if isinstance(G, nx.MultiGraph):
            G.edges[u, v, k]["node_penalty"] = float(node_penalty)
        else:
            data["node_penalty"] = float(node_penalty)


def add_risk_total_attributes(
    G: nx.Graph,
    *,
    risk_cfg: RiskConfig = RiskConfig(eta=1.0),
) -> None:
    """
    Adds 'risk_total' to each edge:
      risk_total = seg_risk + eta * node_penalty

    Requires: seg_risk exists; node_penalty exists (0 if no node ids/risks attached).
    """
    for u, v, k, data in _iter_edges(G):
        seg_risk = float(data.get("seg_risk", 0.0))
        node_penalty = float(data.get("node_penalty", 0.0))
        risk_total = seg_risk + risk_cfg.eta * node_penalty

        if isinstance(G, nx.MultiGraph):
            G.edges[u, v, k]["risk_total"] = float(risk_total)
        else:
            data["risk_total"] = float(risk_total)


# End-to-end graph builder

def build_graph_with_risk(
    segments_panel_gdf: gpd.GeoDataFrame,
    *,
    crossings_gdf: Optional[gpd.GeoDataFrame] = None,
    junction_panel_gdf: Optional[gpd.GeoDataFrame] = None,
    graph_cfg: GraphBuildConfig = GraphBuildConfig(),
    node_risk_col: str = "risk_weight_secure",
    node_snap_m: float = 30.0,
    node_risk_fallback_strategy: str = "median",
    node_risk_fallback_default: float = 0.0,
    risk_cfg: RiskConfig = RiskConfig(eta=1.0),
) -> RoutingGraphArtifacts:
    """
    Build the pooled routing graph and attach cost and risk attributes.

      1) Build graph edges from segment geometries and segment routing weights
      2) Optionally attach node_ids to graph nodes by snapping crossings_gdf points
      3) Optionally build node risk lookup from junction_panel_gdf (pooled)
      4) Add node_penalty and risk_total attributes to edges
    """
    G = build_routing_graph(segments_panel_gdf, config=graph_cfg)

    node_risk_lookup: Dict[str, float] = {}
    used_node_risk_col: Optional[str] = None
    notes: list[str] = []

    need_junction = (risk_cfg.eta != 0.0)

    if need_junction:
        if crossings_gdf is None or junction_panel_gdf is None:
            notes.append(
                "junction risk requested (eta!=0) but crossings_gdf/junction_panel_gdf not provided, junction penalty set to 0"
            )
        else:
            _attach_node_ids_to_graph_nodes(
                G,
                crossings_gdf,
                metric_epsg=graph_cfg.metric_epsg,
                node_id_col="node_id",
                max_snap_m=node_snap_m,
            )
            node_risk_lookup = _build_node_risk_lookup(
                junction_panel_gdf,
                node_id_col="node_id",
                node_risk_col=node_risk_col,
                fallback=node_risk_fallback_default,
                fallback_strategy=node_risk_fallback_strategy,
            )
            used_node_risk_col = node_risk_col

            attached = sum(1 for n in G.nodes if "node_id" in G.nodes[n])
            notes.append(f"Attached node_id to {attached}/{G.number_of_nodes()} graph nodes (snap<= {node_snap_m}m).")

    add_node_penalty_attributes(G, node_risk_by_node_id=node_risk_lookup)
    add_risk_total_attributes(G, risk_cfg=risk_cfg)

    seg_risk_col_used = None
    for c in graph_cfg.seg_risk_col_candidates:
        if c in segments_panel_gdf.columns:
            seg_risk_col_used = c
            break
    if seg_risk_col_used is None:
        seg_risk_col_used = "(unknown)"

    if risk_cfg.eta != 0.0:
        notes.append(f"Risk objective uses eta={risk_cfg.eta} for junction penalty weighting.")

    return RoutingGraphArtifacts(
        G=G,
        segment_risk_col_used=seg_risk_col_used,
        node_risk_col_used=used_node_risk_col,
        notes=" ".join(notes).strip(),
    )


# Verification utilities

def verify_graph_sanity(
    artifacts: RoutingGraphArtifacts,
    *,
    expect_junction_penalties: bool = False,
) -> Dict[str, Union[int, float, str]]:
    """
    Quick sanity checks:
      - graph non-empty
      - risk magnitude reasonable
      - node_penalty and risk_total ranges
      - node ids attached when expected
    """
    G = artifacts.G

    seg_risks = [float(d.get("seg_risk", 0.0)) for _, _, _, d in _iter_edges(G)]
    risk_totals = [float(d.get("risk_total", 0.0)) for _, _, _, d in _iter_edges(G)]
    lengths = [float(d.get("length_m", 0.0)) for _, _, _, d in _iter_edges(G)]
    node_pen = [float(d.get("node_penalty", 0.0)) for _, _, _, d in _iter_edges(G)]

    attached_nodes = sum(1 for n in G.nodes if "node_id" in G.nodes[n])

    out: Dict[str, Union[int, float, str]] = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "seg_risk_min": float(np.min(seg_risks)) if seg_risks else float("nan"),
        "seg_risk_median": float(np.median(seg_risks)) if seg_risks else float("nan"),
        "seg_risk_max": float(np.max(seg_risks)) if seg_risks else float("nan"),
        "risk_total_median": float(np.median(risk_totals)) if risk_totals else float("nan"),
        "risk_total_max": float(np.max(risk_totals)) if risk_totals else float("nan"),
        "length_median": float(np.median(lengths)) if lengths else float("nan"),
        "node_penalty_nonzero_share": float(np.mean(np.array(node_pen) > 0)) if node_pen else 0.0,
        "graph_node_ids_attached": attached_nodes,
        "notes": artifacts.notes,
    }

    if expect_junction_penalties and attached_nodes == 0:
        out["warning"] = (
            "Expected junction penalties, but no graph nodes have node_id attached. "
            "Check snapping tolerance and CRS."
        )
    return out
