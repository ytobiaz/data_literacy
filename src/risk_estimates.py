import geopandas as gpd
import numpy as np
import folium
from branca import colormap as cm
from scipy.stats import gamma


def zero_trip_accident_map_by_year(segments_df, junction_df, year, *, coordinates=(52.518589, 13.376665), zoom=12):
    # yearly aggregation
    seg_year = (
        segments_df[segments_df["year"] == year]
        .groupby("counter_name", as_index=False)
        .agg(geometry=("geometry", "first"),
             trips=("monthly_strava_trips", "sum"),
             accidents=("total_accidents", "sum"))
    )
    seg_year = gpd.GeoDataFrame(seg_year, geometry="geometry", crs=segments_df.crs)

    jn_year = (
        junction_df[junction_df["year"] == year]
        .groupby("node_id", as_index=False)
        .agg(geometry=("geometry", "first"),
             trips=("monthly_strava_trips", "sum"),
             accidents=("total_accidents", "sum"))
    )
    jn_year = gpd.GeoDataFrame(jn_year, geometry="geometry", crs=junction_df.crs)

    seg_zero = seg_year[(seg_year["trips"] == 0) & (seg_year["accidents"] > 0)]
    jn_zero = jn_year[(jn_year["trips"] == 0) & (jn_year["accidents"] > 0)]

    m = folium.Map(location=coordinates, zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
        attr="©CartoDB",
        name="CartoDB Light No Labels",
    ).add_to(m)

    if not seg_zero.empty:
        folium.GeoJson(
            seg_zero[["counter_name", "trips", "accidents", "geometry"]],
            name=f"Segments trips=0 & accidents>0 ({year})",
            style_function=lambda feat: {"color": "red", "weight": 5, "opacity": 0.9},
            tooltip=folium.GeoJsonTooltip(
                fields=["counter_name", "trips", "accidents"],
                aliases=["Segment", "Trips (year sum)", "Accidents (year sum)"],
                localize=True,
            ),
        ).add_to(m)

    if not jn_zero.empty:
        for _, r in jn_zero.iterrows():
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=4,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.9,
                tooltip=f"Junction {int(r['node_id'])} | trips=0 | accidents={int(r['accidents'])}",
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m

def total_relative_risk_map_log(
    segments_df,
    junction_df,
    *,
    coordinates=(52.518589, 13.376665),
    zoom=12,
    segment_weight=5,
    junction_radius=4,
):
    # aggregate over all years
    seg_rr = (
        segments_df.groupby("counter_name", as_index=False)
        .agg(
            geometry=("geometry", "first"),
            total_accidents=("total_accidents", "sum"),
            expected_accidents=("expected_accidents", "sum"),
        )
    )
    seg_rr = gpd.GeoDataFrame(seg_rr, geometry="geometry", crs=segments_df.crs)

    jn_rr = (
        junction_df.groupby("node_id", as_index=False)
        .agg(
            geometry=("geometry", "first"),
            total_accidents=("total_accidents", "sum"),
            expected_accidents=("expected_accidents", "sum"),
        )
    )
    jn_rr = gpd.GeoDataFrame(jn_rr, geometry="geometry", crs=junction_df.crs)

    # compute relative risk
    seg_rr["relative_risk"] = seg_rr["total_accidents"] / seg_rr["expected_accidents"]
    jn_rr["relative_risk"] = jn_rr["total_accidents"] / jn_rr["expected_accidents"]

    # compute scale (vabs) from valid positives only
    seg_valid = seg_rr["relative_risk"].replace([np.inf, -np.inf], np.nan)
    jn_valid = jn_rr["relative_risk"].replace([np.inf, -np.inf], np.nan)
    all_valid = np.concatenate([
        seg_valid[seg_valid > 0].to_numpy(),
        jn_valid[jn_valid > 0].to_numpy(),
    ])
    vabs = np.nanpercentile(np.abs(np.log10(all_valid)), 99) if all_valid.size else 1.0
    vabs = max(vabs, 0.1)

    # floor tied to scale
    floor = 10 ** (-vabs)

    # set 0/NaN/inf to floor, then log
    seg_rr["relative_risk"] = seg_rr["relative_risk"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    jn_rr["relative_risk"] = jn_rr["relative_risk"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    seg_rr["relative_risk"] = seg_rr["relative_risk"].clip(lower=floor)
    jn_rr["relative_risk"] = jn_rr["relative_risk"].clip(lower=floor)

    seg_rr["log_rr"] = np.log10(seg_rr["relative_risk"])
    jn_rr["log_rr"] = np.log10(jn_rr["relative_risk"])

    # diverging colormap: green -> yellow -> red
    colormap = cm.LinearColormap(
        colors=["green", "yellow", "red"],
        vmin=-vabs,
        vmax=vabs,
        index=[-vabs, 0.0, vabs],
    )
    colormap.caption = "log10(relative risk) | 0 = RR 1.0"

    m = folium.Map(location=coordinates, zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
        attr="©CartoDB",
    ).add_to(m)

    if not seg_rr.empty:
        def seg_style_fn(feat):
            lr = float(feat["properties"]["log_rr"])
            lr = max(min(lr, vabs), -vabs)
            return {"color": colormap(lr), "weight": segment_weight, "opacity": 0.9}

        folium.GeoJson(
            seg_rr[["counter_name", "log_rr", "relative_risk", "geometry"]],
            name="Segments (log10 RR)",
            style_function=seg_style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["counter_name", "relative_risk"],
                aliases=["Segment", "Relative risk"],
                localize=True,
            ),
        ).add_to(m)

    if not jn_rr.empty:
        for _, r in jn_rr.iterrows():
            lr = max(min(float(r["log_rr"]), vabs), -vabs)
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=junction_radius,
                color=colormap(lr),
                fill=True,
                fill_color=colormap(lr),
                fill_opacity=0.9,
                tooltip=f"Junction {int(r['node_id'])} | RR={r['relative_risk']:.2f}",
            ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


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

def total_relative_risk_map_log_eb_sig(
    segments_df,
    junction_df,
    alpha,
    *,
    coordinates=(52.518589, 13.376665),
    zoom=12,
    segment_weight=5,
    junction_radius=4,
    ci=0.95
):

    # aggregate over all years
    seg_agg = {
        "geometry": ("geometry", "first"),
        "total_accidents": ("total_accidents", "sum"),
        "expected_accidents": ("expected_accidents", "sum"),
        "total_trips": ("monthly_strava_trips", "sum"),
    }

    seg_rr = (
        segments_df.groupby("counter_name", as_index=False)
        .agg(**seg_agg)
    )
    seg_rr = gpd.GeoDataFrame(seg_rr, geometry="geometry", crs=segments_df.crs)

    jn_rr = (
        junction_df.groupby("node_id", as_index=False)
        .agg(
            geometry=("geometry", "first"),
            total_accidents=("total_accidents", "sum"),
            expected_accidents=("expected_accidents", "sum"),
            total_trips=("monthly_strava_trips", "sum"),
        )
    )
    jn_rr = gpd.GeoDataFrame(jn_rr, geometry="geometry", crs=junction_df.crs)

    # EB-smoothed relative risk
    seg_rr["relative_risk"] = (seg_rr["total_accidents"] + alpha) / (seg_rr["expected_accidents"] + alpha)
    jn_rr["relative_risk"] = (jn_rr["total_accidents"] + alpha) / (jn_rr["expected_accidents"] + alpha)

    # EB CIs
    q_lo = (1.0 - ci) / 2.0
    q_hi = 1.0 - q_lo

    seg_a = alpha + seg_rr["total_accidents"]
    seg_rate = alpha + seg_rr["expected_accidents"]
    seg_rr["rr_lo"] = gamma.ppf(q_lo, a=seg_a, scale=1.0 / seg_rate)
    seg_rr["rr_hi"] = gamma.ppf(q_hi, a=seg_a, scale=1.0 / seg_rate)

    jn_a = alpha + jn_rr["total_accidents"]
    jn_rate = alpha + jn_rr["expected_accidents"]
    jn_rr["rr_lo"] = gamma.ppf(q_lo, a=jn_a, scale=1.0 / jn_rate)
    jn_rr["rr_hi"] = gamma.ppf(q_hi, a=jn_a, scale=1.0 / jn_rate)

    # significance vs 1
    seg_rr["sig"] = np.where(seg_rr["rr_lo"] > 1.0, "above",
                      np.where(seg_rr["rr_hi"] < 1.0, "below", "ns"))
    jn_rr["sig"] = np.where(jn_rr["rr_lo"] > 1.0, "above",
                      np.where(jn_rr["rr_hi"] < 1.0, "below", "ns"))

    # compute scale (vabs) from valid positives only
    seg_valid = seg_rr["relative_risk"].replace([np.inf, -np.inf], np.nan)
    jn_valid = jn_rr["relative_risk"].replace([np.inf, -np.inf], np.nan)
    all_valid = np.concatenate([
        seg_valid[seg_valid > 0].to_numpy(),
        jn_valid[jn_valid > 0].to_numpy(),
    ])
    vabs = np.nanpercentile(np.abs(np.log10(all_valid)), 99) if all_valid.size else 1.0
    vabs = max(vabs, 0.1)

    floor = 10 ** (-vabs)

    seg_rr["relative_risk"] = seg_rr["relative_risk"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    jn_rr["relative_risk"] = jn_rr["relative_risk"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    seg_rr["relative_risk"] = seg_rr["relative_risk"].clip(lower=floor)
    jn_rr["relative_risk"] = jn_rr["relative_risk"].clip(lower=floor)

    seg_rr["log_rr"] = np.log10(seg_rr["relative_risk"])
    jn_rr["log_rr"] = np.log10(jn_rr["relative_risk"])

    # diverging colormap: green -> yellow -> red
    colormap = cm.LinearColormap(
        colors=["green", "yellow", "red"],
        vmin=-vabs,
        vmax=vabs,
        index=[-vabs, 0.0, vabs],
    )
    colormap.caption = "log10(EB-smoothed RR) | 0 = RR 1.0"

    m = folium.Map(location=coordinates, zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
        attr="©CartoDB",
    ).add_to(m)

    if not seg_rr.empty:
        def seg_style_fn(feat):
            sig = feat["properties"]["sig"]
            if sig == "below":
                rr_for_color = float(feat["properties"]["rr_hi"])  # upper limit (<1)
            elif sig == "above":
                rr_for_color = float(feat["properties"]["rr_lo"])  # lower limit (>1)
            else:
                rr_for_color = 1.0  # not significant

            lr = np.log10(rr_for_color)
            lr = max(min(lr, vabs), -vabs)
            return {"color": colormap(lr), "weight": segment_weight, "opacity": 0.9}

        seg_fields = ["counter_name", "relative_risk", "rr_lo", "rr_hi", "sig", "total_accidents", "total_trips"]
        seg_aliases = ["Segment", "EB RR", "CI lo", "CI hi", "Significance", "Accidents", "Strava trips"]

        folium.GeoJson(
            seg_rr[seg_fields + ["log_rr", "geometry"]],
            name="Segments (EB log10 RR)",
            style_function=seg_style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=seg_fields,
                aliases=seg_aliases,
                localize=True,
            ),
        ).add_to(m)

    if not jn_rr.empty:
        for _, r in jn_rr.iterrows():
            if r["sig"] == "below":
                rr_for_color = r["rr_hi"]
            elif r["sig"] == "above":
                rr_for_color = r["rr_lo"]
            else:
                rr_for_color = 1.0

            lr = np.log10(rr_for_color)
            lr = max(min(lr, vabs), -vabs)
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=junction_radius,
                color=colormap(lr),
                fill=True,
                fill_color=colormap(lr),
                fill_opacity=0.9,
                tooltip=(
                    f"Junction {int(r['node_id'])} | EB RR={r['relative_risk']:.2f} "
                    f"[{r['rr_lo']:.2f}, {r['rr_hi']:.2f}] | {r['sig']} | "
                    f"Accidents={int(r['total_accidents'])} | Trips={int(r['total_trips'])}"
                ),
            ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

def total_relative_risk_map_log_eb(
    segments_df,
    junction_df,
    alpha,
    *,
    coordinates=(52.518589, 13.376665),
    zoom=12,
    segment_weight=5,
    junction_radius=4,
):
    # aggregate over all years
    seg_rr = (
        segments_df.groupby("counter_name", as_index=False)
        .agg(
            geometry=("geometry", "first"),
            total_accidents=("total_accidents", "sum"),
            expected_accidents=("expected_accidents", "sum"),
        )
    )
    seg_rr = gpd.GeoDataFrame(seg_rr, geometry="geometry", crs=segments_df.crs)

    jn_rr = (
        junction_df.groupby("node_id", as_index=False)
        .agg(
            geometry=("geometry", "first"),
            total_accidents=("total_accidents", "sum"),
            expected_accidents=("expected_accidents", "sum"),
        )
    )
    jn_rr = gpd.GeoDataFrame(jn_rr, geometry="geometry", crs=junction_df.crs)

    # EB-smoothed relative risk
    seg_rr["relative_risk"] = (seg_rr["total_accidents"] + alpha) / (seg_rr["expected_accidents"] + alpha)
    jn_rr["relative_risk"] = (jn_rr["total_accidents"] + alpha) / (jn_rr["expected_accidents"] + alpha)

    # compute scale (vabs) from valid positives only
    seg_valid = seg_rr["relative_risk"].replace([np.inf, -np.inf], np.nan)
    jn_valid = jn_rr["relative_risk"].replace([np.inf, -np.inf], np.nan)
    all_valid = np.concatenate([
        seg_valid[seg_valid > 0].to_numpy(),
        jn_valid[jn_valid > 0].to_numpy(),
    ])
    vabs = np.nanpercentile(np.abs(np.log10(all_valid)), 99) if all_valid.size else 1.0
    vabs = max(vabs, 0.1)

    # floor tied to scale
    floor = 10 ** (-vabs)

    # set 0/NaN/inf to floor, then log
    seg_rr["relative_risk"] = seg_rr["relative_risk"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    jn_rr["relative_risk"] = jn_rr["relative_risk"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    seg_rr["relative_risk"] = seg_rr["relative_risk"].clip(lower=floor)
    jn_rr["relative_risk"] = jn_rr["relative_risk"].clip(lower=floor)

    seg_rr["log_rr"] = np.log10(seg_rr["relative_risk"])
    jn_rr["log_rr"] = np.log10(jn_rr["relative_risk"])

    # diverging colormap: green -> yellow -> red
    colormap = cm.LinearColormap(
        colors=["green", "yellow", "red"],
        vmin=-vabs,
        vmax=vabs,
        index=[-vabs, 0.0, vabs],
    )
    colormap.caption = "log10(EB-smoothed RR) | 0 = RR 1.0"

    m = folium.Map(location=coordinates, zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
        attr="©CartoDB",
    ).add_to(m)

    if not seg_rr.empty:
        def seg_style_fn(feat):
            lr = float(feat["properties"]["log_rr"])
            lr = max(min(lr, vabs), -vabs)
            return {"color": colormap(lr), "weight": segment_weight, "opacity": 0.9}

        folium.GeoJson(
            seg_rr[["counter_name", "log_rr", "relative_risk", "geometry"]],
            name="Segments (EB log10 RR)",
            style_function=seg_style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["counter_name", "relative_risk"],
                aliases=["Segment", "EB RR"],
                localize=True,
            ),
        ).add_to(m)

    if not jn_rr.empty:
        for _, r in jn_rr.iterrows():
            lr = max(min(float(r["log_rr"]), vabs), -vabs)
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=junction_radius,
                color=colormap(lr),
                fill=True,
                fill_color=colormap(lr),
                fill_opacity=0.9,
                tooltip=f"Junction {int(r['node_id'])} | EB RR={r['relative_risk']:.2f}",
            ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m