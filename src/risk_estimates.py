import geopandas as gpd
import numpy as np
import folium
from branca import colormap as cm
from scipy.stats import gamma


def ensure_active_geometry(gdf: gpd.GeoDataFrame, geometry_col: str = "geometry") -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with a valid active geometry column.

    Groupby/agg operations in pandas can drop the active geometry metadata
    (e.g., leaving the active geometry name as "0"). This helper restores a
    consistent active geometry column so downstream geospatial methods like
    ``to_crs`` work reliably.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry=geometry_col)

    # If the active geometry is missing but a geometry-typed column exists,
    # make it active again.
    if geometry_col in gdf.columns:
        gdf = gdf.set_geometry(geometry_col)

    return gdf


def aggregate_junction_points(junction_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Aggregate to one point per junction while preserving geometry/CRS."""
    j_plot = junction_df[junction_df.geometry.notna()].copy()

    j_points = (
        j_plot.groupby("node_id", as_index=False)
        .agg(
            geometry=("geometry", "first"),
            trips=("monthly_strava_trips", "sum"),
            accidents=("total_accidents", "sum"),
        )
    )

    return gpd.GeoDataFrame(j_points, geometry="geometry", crs=j_plot.crs)



def estimate_alpha_simple_moments(df, y_col, yhat_col, *, alpha_min=1e-6, alpha_max=1e6):
    # Use only rows with positive expected counts
    valid = df[df[yhat_col] > 0].copy()
    if valid.empty:
        return alpha_max

    y = valid[y_col].to_numpy()
    yhat = valid[yhat_col].to_numpy()
    weights = yhat
    term = ((y - yhat) ** 2 - y) / (yhat ** 2)
    m = np.average(term, weights=weights)

    if m <= 0 or not np.isfinite(m):
        alpha = alpha_max  # ~ no shrinkage
    else:
        alpha = 1.0 / m
        alpha = float(np.clip(alpha, alpha_min, alpha_max))

    return alpha



def plot_rr_map(
    segments_df,
    junction_df,
    *,
    show="both",  # "both" | "segments" | "junctions"
    metric="raw_rr",  # "raw_rr" | any column name
    seg_metric_col=None,
    jn_metric_col=None,
    zoom=12,
    coordinates=None,
    segment_weight=4,
    junction_radius=4,
    clip_percentile=99,
):
    import geopandas as gpd
    import numpy as np
    import folium
    from branca import colormap as cm

    def _ensure_geo(gdf, geometry_col="geometry"):
        if not isinstance(gdf, gpd.GeoDataFrame):
            gdf = gpd.GeoDataFrame(gdf, geometry=geometry_col)
        if geometry_col in gdf.columns:
            gdf = gdf.set_geometry(geometry_col)
        return gdf

    def _aggregate(df, id_col, extra_cols=()):
        agg_spec = dict(
            geometry=("geometry", "first"),
            total_accidents=("total_accidents", "sum"),
        )

        if "sum_strava_total_trip_count" in df.columns:
            agg_spec["exposure"] = ("sum_strava_total_trip_count", "sum")
        elif "monthly_strava_trips" in df.columns:
            agg_spec["exposure"] = ("monthly_strava_trips", "sum")
        elif "exposure" in df.columns:
            agg_spec["exposure"] = ("exposure", "sum")

        for c in extra_cols:
            if c and c in df.columns:
                agg_spec[c] = (c, "first")
        out = df.groupby(id_col, as_index=False).agg(**agg_spec)
        return gpd.GeoDataFrame(out, geometry="geometry", crs=df.crs)

    def _resolve_metric(df, kind, metric_name, metric_col):
        if metric_col is not None:
            if metric_col not in df.columns:
                raise KeyError(f"{kind}: metric column {metric_col!r} not in dataframe")
            return metric_col

        if metric_name == "raw_rr" and "raw_rr" not in df.columns:
            raise KeyError(
                f"{kind}: metric 'raw_rr' not found. "
                f"Pass seg_metric_col/jn_metric_col or compute it before plotting."
            )

        if metric_name not in df.columns:
            raise KeyError(
                f"{kind}: metric {metric_name!r} not found. "
                f"Pass seg_metric_col/jn_metric_col or ensure it exists before aggregation."
            )
        return metric_name

    if show not in {"both", "segments", "junctions"}:
        raise ValueError("show must be one of: 'both', 'segments', 'junctions'")

    segments_df = _ensure_geo(segments_df)
    junction_df = _ensure_geo(junction_df)

    seg = _aggregate(segments_df, "counter_name", extra_cols=[metric, seg_metric_col]) if show in {"both", "segments"} else None
    jn = _aggregate(junction_df, "node_id", extra_cols=[metric, jn_metric_col]) if show in {"both", "junctions"} else None

    seg_metric = None
    if seg is not None:
        seg_metric = _resolve_metric(seg, "segments", metric, seg_metric_col)
        seg[seg_metric] = seg[seg_metric].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    jn_metric = None
    if jn is not None:
        jn_metric = _resolve_metric(jn, "junctions", metric, jn_metric_col)
        jn[jn_metric] = jn[jn_metric].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vals = []
    if seg is not None:
        vals.append(seg.loc[seg[seg_metric] > 0, seg_metric].to_numpy())
    if jn is not None:
        vals.append(jn.loc[jn[jn_metric] > 0, jn_metric].to_numpy())
    all_pos = np.concatenate([v for v in vals if v.size]) if vals else np.array([])

    vabs = float(np.nanpercentile(np.abs(np.log10(all_pos)), clip_percentile)) if all_pos.size else 1.0
    vabs = max(vabs, 0.1)
    floor = 10 ** (-vabs)

    if seg is not None:
        seg["_metric"] = seg[seg_metric].clip(lower=floor)
        seg["_log_rr"] = np.log10(seg["_metric"])
        seg_wgs = seg.to_crs(epsg=4326)
    else:
        seg_wgs = None

    if jn is not None:
        jn["_metric"] = jn[jn_metric].clip(lower=floor)
        jn["_log_rr"] = np.log10(jn["_metric"])
        jn_wgs = jn.to_crs(epsg=4326)
    else:
        jn_wgs = None

    if coordinates is not None:
        center = coordinates
    elif jn_wgs is not None and len(jn_wgs):
        center = [float(jn_wgs.geometry.y.mean()), float(jn_wgs.geometry.x.mean())]
    elif seg_wgs is not None and len(seg_wgs):
        center = [float(seg_wgs.geometry.centroid.y.mean()), float(seg_wgs.geometry.centroid.x.mean())]
    else:
        center = (52.518589, 13.376665)

    cmap = cm.LinearColormap(
        colors=["green", "yellow", "red"],
        vmin=-vabs,
        vmax=vabs,
        index=[-vabs, 0.0, vabs],
    )
    cmap.caption = f"log10({seg_metric or jn_metric or metric})"

    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    if seg_wgs is not None and len(seg_wgs):
        seg_fields = ["counter_name", seg_metric]
        seg_aliases = ["Segment", seg_metric]
        if "exposure" in seg_wgs.columns:
            seg_fields.append("exposure")
            seg_aliases.append("Exposure")
        if "total_accidents" in seg_wgs.columns:
            seg_fields.append("total_accidents")
            seg_aliases.append("Accidents")

        def _seg_style(feat):
            lr = float(feat["properties"]["_log_rr"])
            lr = max(min(lr, vabs), -vabs)
            return {"color": cmap(lr), "weight": segment_weight, "opacity": 0.85}

        folium.GeoJson(
            seg_wgs[seg_fields + ["_log_rr", "geometry"]],
            name=f"Segments ({seg_metric})",
            style_function=_seg_style,
            tooltip=folium.GeoJsonTooltip(
                fields=seg_fields,
                aliases=seg_aliases,
                localize=True,
            ),
        ).add_to(m)

    if jn_wgs is not None and len(jn_wgs):
        for _, r in jn_wgs.iterrows():
            if r.geometry is None or not np.isfinite(r.geometry.x) or not np.isfinite(r.geometry.y):
                continue
            lr = max(min(float(r["_log_rr"]), vabs), -vabs)
            exposure_txt = f" | exposure={float(r['exposure']):.0f}" if "exposure" in jn_wgs.columns else ""
            accidents_txt = (
                f" | accidents={int(r['total_accidents'])}"
                if "total_accidents" in jn_wgs.columns and np.isfinite(r["total_accidents"])
                else ""
            )
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=junction_radius,
                color=cmap(lr),
                fill=True,
                fill_color=cmap(lr),
                fill_opacity=0.9,
                weight=1,
                tooltip=(
                    f"node_id={int(r.node_id)} | {jn_metric}={float(r[jn_metric]):.2f}"
                    f"{exposure_txt}{accidents_txt}"
                ),
            ).add_to(m)

    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

def add_secure_rr(df, rr="rr_eb", lo="rr_eb_lo", hi="rr_eb_hi", out="rr_secure"):
    sig_below = df[hi] < 1.0
    sig_above = df[lo] > 1.0
    df[out] = np.select(
        [sig_below, sig_above],
        [df[hi], df[lo]],
        default=1.0,
    )
    return df


def plot_rr_map_with_accidents(
    top_segments_df,
    top_junctions_df,
    accidents_df,
    *,
    metric="rr_secure",
    year_col="year",
    severity_col="injury_severity",
    lon_col="XGCSWGS84",
    lat_col="YGCSWGS84",
    zoom=12,
    coordinates=None,
    segment_weight=4,
    junction_radius=4,
    clip_percentile=99,
):
    import geopandas as gpd
    import numpy as np
    import folium
    import pandas as pd

    m = plot_rr_map(
        top_segments_df,
        top_junctions_df,
        show="both",
        metric=metric,
        zoom=zoom,
        coordinates=coordinates,
        segment_weight=segment_weight,
        junction_radius=junction_radius,
        clip_percentile=clip_percentile,
    )

    if not isinstance(accidents_df, gpd.GeoDataFrame) or accidents_df.geometry.name not in accidents_df.columns:
        if lon_col not in accidents_df.columns or lat_col not in accidents_df.columns:
            raise KeyError(f"Accidents dataframe must have geometry or {lon_col!r}/{lat_col!r} columns.")

        acc_df = accidents_df.copy()
        acc_df[lon_col] = pd.to_numeric(acc_df[lon_col], errors="coerce")
        acc_df[lat_col] = pd.to_numeric(acc_df[lat_col], errors="coerce")
        acc_df = acc_df.dropna(subset=[lon_col, lat_col])

        accidents_gdf = gpd.GeoDataFrame(
            acc_df,
            geometry=gpd.points_from_xy(acc_df[lon_col], acc_df[lat_col]),
            crs="EPSG:4326",
        )
    else:
        accidents_gdf = accidents_df.copy()

    required_cols = {year_col, severity_col}
    missing = required_cols - set(accidents_gdf.columns)
    if missing:
        raise KeyError(f"Accidents dataframe missing required columns: {sorted(missing)}")

    accidents_gdf = accidents_gdf[accidents_gdf.geometry.notna()].copy()

    years = sorted(accidents_gdf[year_col].dropna().unique().tolist())
    severities = sorted(accidents_gdf[severity_col].dropna().unique().tolist())

    for year in years:
        for severity in severities:
            subset = accidents_gdf[
                (accidents_gdf[year_col] == year) & (accidents_gdf[severity_col] == severity)
            ]
            if subset.empty:
                continue
            layer = folium.FeatureGroup(
                name=f"Accidents {year} | severity {severity}",
                show=False,
            )
            for _, r in subset.iterrows():
                if r.geometry is None or not np.isfinite(r.geometry.x) or not np.isfinite(r.geometry.y):
                    continue
                folium.CircleMarker(
                    location=[r.geometry.y, r.geometry.x],
                    radius=2,
                    color="#333333",
                    fill=True,
                    fill_color="#333333",
                    fill_opacity=0.6,
                    weight=0,
                    tooltip=f"{year_col}={year} | {severity_col}={severity}",
                ).add_to(layer)
            layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
