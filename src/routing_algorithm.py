"""
Routing algorithms for shortest path and constrained min-risk routing.

This module provides algorithms for computing routes on a graph with risk attributes:
- Shortest path by distance or risk
- Constrained min-risk routing (minimize risk subject to distance constraint)
- Origin-destination (OD) routing wrapper
- Route statistics computation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point

from .routing_graph import (
    GraphBuildConfig,
    NodeKey,
    RiskConfig,
    RoutingGraphArtifacts,
    _get_edge_data_for_step,
    _iter_edges,
    build_graph_with_risk,
    verify_graph_sanity,
)


# Routing utilities

def nearest_graph_node(
    G: nx.Graph,
    lon: float,
    lat: float,
    *,
    metric_epsg: int = 32633,
) -> NodeKey:
    """Find the nearest graph node to a geographic point (lon, lat) using brute-force distance calculation."""
    p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=metric_epsg).iloc[0]
    x, y = float(p.x), float(p.y)

    nodes = list(G.nodes())
    if len(nodes) == 0:
        raise ValueError("Graph has no nodes")

    xy = np.array(nodes, dtype=float)
    d2 = (xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2
    idx = int(d2.argmin())
    return nodes[idx]


def route_stats(G: nx.Graph, path: List[NodeKey], *, choose_by: str = "length_m") -> Dict[str, float]:
    """
    Summarize route length and risk along a node-path.

    For MultiGraph, the parallel edge between consecutive nodes is chosen by
    minimizing choose_by. This MUST match the weight used when the path was computed,
    otherwise stats can be inconsistent.
    """
    total_len = 0.0
    total_seg_risk = 0.0
    total_node_penalty = 0.0
    total_risk_total = 0.0

    for a, b in zip(path[:-1], path[1:]):
        d = _get_edge_data_for_step(G, a, b, choose_by=choose_by)

        total_len += float(d.get("length_m", 0.0))
        total_seg_risk += float(d.get("seg_risk", 0.0))
        total_node_penalty += float(d.get("node_penalty", 0.0))
        total_risk_total += float(d.get("risk_total", d.get("seg_risk", 0.0)))

    return {
        "length_m": total_len,
        "seg_risk_sum": total_seg_risk,
        "node_penalty_sum": total_node_penalty,
        "risk_total_sum": total_risk_total,
    }


def shortest_path_by(G: nx.Graph, source: NodeKey, target: NodeKey, weight: str) -> Optional[List[NodeKey]]:
    """Compute shortest path by given weight attribute."""
    try:
        return nx.shortest_path(G, source=source, target=target, weight=weight, method="dijkstra")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def constrained_min_risk_route(
    G: nx.Graph,
    source: NodeKey,
    target: NodeKey,
    *,
    eps: float = 0.10,                 # allow up to +10% distance over shortest
    length_attr: str = "length_m",
    risk_attr: str = "risk_total",
    lambdas: Optional[List[float]] = None,
) -> Optional[List[NodeKey]]:
    """
    Minimize risk subject to length <= (1+eps) * shortest_length.

    Implementation: parametric sweep with a weighted sum:
      minimize (risk + lambda * length)
    and select the best path that satisfies the length constraint.
    """
    shortest_len_path = shortest_path_by(G, source, target, weight=length_attr)
    if shortest_len_path is None:
        return None  # disconnected

    shortest_len = route_stats(G, shortest_len_path, choose_by=length_attr)["length_m"]
    max_len = (1.0 + eps) * shortest_len

    if lambdas is None:
        lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

    best_path: Optional[List[NodeKey]] = None
    best_risk: Optional[float] = None
    best_len: Optional[float] = None

    for lam in lambdas:
        # Build temporary combined weight on edges: risk + lambda*length
        for u, v, k, d in _iter_edges(G):
            comb = float(d.get(risk_attr, 0.0)) + lam * float(d.get(length_attr, 0.0))
            if isinstance(G, nx.MultiGraph):
                G.edges[u, v, k]["_comb"] = comb
            else:
                d["_comb"] = comb

        p = shortest_path_by(G, source, target, weight="_comb")
        if p is None:
            continue

        st = route_stats(G, p, choose_by="_comb")  # consistent parallel-edge choice

        if st["length_m"] <= max_len:
            candidate_risk = st["risk_total_sum"]  
            if (
                best_path is None
                or candidate_risk < best_risk
                or (candidate_risk == best_risk and st["length_m"] < best_len)
            ):
                best_path = p
                best_risk = candidate_risk
                best_len = st["length_m"]

    return best_path if best_path is not None else shortest_len_path


# Visualization helpers

def _flatten_lines(geom):
    """Extract LineStrings from geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    try:
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    except Exception:
        return []


def path_to_multiline_latlon(
    G: nx.Graph,
    path_nodes: Optional[List[NodeKey]],
    *,
    metric_epsg: int = 32633,
    choose_by: str = "length_m",
):
    """Convert a node path to a MultiLineString geometry in lat/lon coordinates (EPSG:4326) for visualization."""
    if path_nodes is None or len(path_nodes) < 2:
        return None

    geoms = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if isinstance(G, nx.MultiGraph):
            edges = G.get_edge_data(u, v) or {}
            if not edges:
                continue
            best_key = min(
                edges,
                key=lambda k: float(edges[k].get(choose_by, edges[k].get("length_m", float("inf"))))
            )
            edge = edges[best_key]
        else:
            edge = G.get_edge_data(u, v) or {}

        geoms.extend(_flatten_lines(edge.get("geometry")))

    if not geoms:
        return None

    ml = MultiLineString(geoms)
    return gpd.GeoSeries([ml], crs=f"EPSG:{metric_epsg}").to_crs(epsg=4326).iloc[0]


# OD routing wrapper

def run_one_od_routing(
    *,
    segments_panel_gdf: gpd.GeoDataFrame,
    crossings_gdf: gpd.GeoDataFrame,
    junction_panel_gdf: gpd.GeoDataFrame,
    origin_lonlat: Tuple[float, float],
    dest_lonlat: Tuple[float, float],
    eps: float = 0.10,
    eta: float = 1.0,
    metric_epsg: int = 32633,
) -> Dict:
    """
    Compute origin-destination routing: shortest path by distance and constrained minimum-risk route.
    
    Returns paths, statistics, and comparison metrics (delta_L, delta_R) between distance-optimal and risk-minimizing routes.
    """
    artifacts = build_graph_with_risk(
        segments_panel_gdf,
        crossings_gdf=crossings_gdf,
        junction_panel_gdf=junction_panel_gdf,
        graph_cfg=GraphBuildConfig(metric_epsg=metric_epsg),
        risk_cfg=RiskConfig(eta=eta),
        node_snap_m=20.0,
    )

    sanity = verify_graph_sanity(
        artifacts,
        expect_junction_penalties=(eta != 0.0),
    )

    G = artifacts.G
    o_lon, o_lat = origin_lonlat
    d_lon, d_lat = dest_lonlat

    src = nearest_graph_node(G, o_lon, o_lat, metric_epsg=metric_epsg)
    dst = nearest_graph_node(G, d_lon, d_lat, metric_epsg=metric_epsg)

    # Baseline: shortest distance (P_dist)
    p_len = shortest_path_by(G, src, dst, weight="length_m")
    if p_len is None:
        return {
            "status": "disconnected",
            "graph_sanity": sanity,
            "origin_node": src,
            "dest_node": dst,
            "notes": artifacts.notes,
        }
    st_len = route_stats(G, p_len, choose_by="length_m")

    # Constrained min-risk (P_safe) minimizing risk_total, with detour constraint
    p_safe = constrained_min_risk_route(
        G,
        src,
        dst,
        eps=eps,
        length_attr="length_m",
        risk_attr="risk_total",
    )
    # If graph is connected, p_safe will at least fall back to p_len (see routing util)
    st_safe = route_stats(G, p_safe, choose_by="risk_total") if p_safe is not None else None

    
    # Compute deltas 
    delta_L = (st_safe["length_m"] - st_len["length_m"]) / st_len["length_m"] if st_safe else None
    if st_len["risk_total_sum"] > 0 and st_safe is not None:
        delta_R = (st_len["risk_total_sum"] - st_safe["risk_total_sum"]) / st_len["risk_total_sum"]
    else:
        delta_R = None

    return {
        "status": "ok",
        "graph_sanity": sanity,
        "origin_node": src,
        "dest_node": dst,
        "shortest_length_path": p_len,
        "shortest_length_stats": st_len,
        "constrained_min_risk_path": p_safe,
        "constrained_min_risk_stats": st_safe,
        "delta_L": delta_L,
        "delta_R": delta_R,
        "notes": artifacts.notes,
        "params": {
            "eps": eps,
            "eta": eta,
        },
        "graph": G,
    }
