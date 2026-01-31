from __future__ import annotations

from typing import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

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
        "road_condition"
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
    trip_col: str = "sum_strava_total_trip_count",
    how: str = "outer",
) -> pd.DataFrame:
    """
    Merge exposure and accidents at segment×year×month.

    Args:
        final_exposure_ym: Strava traffic exposure panel (segment × year × month)
        accidents_agg_ym_rich: Aggregated accidents panel (segment × year × month)
        merge_keys: Columns to merge on
        trip_col: Name of trip count column
        how: Type of merge ('left', 'right', 'outer', 'inner'). Default 'outer' shows:
          - accident months with missing exposure rows
          - exposure months with zero accidents

    Returns:
        Merged panel. Downstream, filter to exposure>0 for modeling risk.
    """

    merge_keys = list(merge_keys)

    merged = final_exposure_ym.merge(
        accidents_agg_ym_rich,
        on=merge_keys,
        how=how,
        validate="one_to_one",
    )

    # Track whether exposure row was missing (important for audits)
    if trip_col in merged.columns:
        merged["exposure_row_missing"] = merged[trip_col].isna()
        merged[trip_col] = merged[trip_col].fillna(0)
    else:
        merged["exposure_row_missing"] = True  # conservative

    # Fill accident columns with 0 where missing
    acc_cols = [c for c in accidents_agg_ym_rich.columns if c not in merge_keys]
    for c in acc_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    return merged



def plot_merged_panel_quality_overview(
    merged_panel: pd.DataFrame,
    segment_geometry: gpd.GeoDataFrame | None = None,
    trip_col: str = "sum_strava_total_trip_count",
    accident_col: str = "total_accidents",
    figsize=(14, 10),
    use_tueplots=True,
    save_path: str | Path | None = None,
) -> tuple:
    """
    Create comprehensive quality check visualization for merged accident-exposure panel.
    
    Creates a 2x2 grid with 4 diagnostic plots:
    - Accidents vs Traffic Exposure scatter
    - Segment Totals: Trips vs Accidents
    - Distribution of accidents per segment by month (boxplot)
    - Geospatial map of accidents (if geometry provided)
    
    Displays quality status inline.
    
    Parameters
    ----------
    merged_panel : DataFrame
        Merged segment×year×month panel with traffic and accident data
    segment_geometry : GeoDataFrame, optional
        Segment geometries for mapping accidents, by default None
    trip_col : str, optional
        Name of trip count column, by default "sum_strava_total_trip_count"
    accident_col : str, optional
        Name of accident count column, by default "total_accidents"
    figsize : tuple, optional
        Figure size (width, height), by default (14, 10)
    use_tueplots : bool, optional
        Whether to use tueplots ICML2024 stylesheet, by default True
    save_path : str | Path | None, optional
        Path to save the figure, by default None
    
    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes objects
    """
    # Apply tueplots styling
    if use_tueplots:
        from tueplots import bundles
        from tueplots.constants.color import palettes
        plt.rcParams.update(bundles.icml2024(column="full", nrows=2, ncols=2))
        colors = palettes.tue_plot
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    main_color = colors[0]
    accent_color = colors[1]
    third_color = colors[2]
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten()
    
    # ===== PLOT 1: Accident vs Exposure scatter =====
    ax1 = axes_flat[0]
    has_accidents = merged_panel[accident_col].notna()
    ax1.scatter(
        merged_panel[trip_col],
        merged_panel.loc[has_accidents, accident_col],
        alpha=0.5,
        s=25,
        color=accent_color,
        edgecolors='none'
    )
    ax1.set_title('Accidents vs Traffic Exposure', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Monthly Trips (log scale)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accident Count', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # ===== PLOT 2: Segment-level summary: trips vs accidents =====
    ax2 = axes_flat[1]
    seg_stats = merged_panel.groupby('counter_name').agg({
        trip_col: 'sum',
        accident_col: 'sum'
    }).dropna()
    ax2.scatter(
        seg_stats[trip_col],
        seg_stats[accident_col],
        alpha=0.5,
        s=25,
        color=third_color,
        edgecolors='none'
    )
    ax2.set_title('Segment Totals: Trips vs Accidents', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Total Trips 2019-2023 (log scale)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Total Accidents', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # ===== PLOT 3: Distribution of accidents per segment per month =====
    ax3 = axes_flat[2]
    
    # Calculate all accidents per segment per month (including zeros)
    all_acc_per_seg_month = []
    for month in range(1, 13):
        month_mask = merged_panel['month'] == month
        acc_per_seg = merged_panel[month_mask].groupby('counter_name')[accident_col].sum()
        # Include all segments (including those with 0 accidents)
        all_acc_per_seg_month.extend(acc_per_seg.values)
    
    # Create histogram of accidents per segment per month
    max_acc = int(max(all_acc_per_seg_month)) if all_acc_per_seg_month else 0
    bins = range(0, max_acc + 2)
    ax3.hist(all_acc_per_seg_month, bins=bins, color=main_color, edgecolor='black', linewidth=0.5)
    
    # Calculate statistics for legend
    mean_acc = np.mean(all_acc_per_seg_month)
    median_acc = np.median(all_acc_per_seg_month)
    
    # Add vertical lines for mean and median
    ax3.axvline(mean_acc, color=accent_color, linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.2f}')
    ax3.axvline(median_acc, color=third_color, linestyle='--', linewidth=2, label=f'Median: {median_acc:.2f}')
    
    ax3.set_title('Distribution: Accidents per Segment per Month', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Accident Count (per Segment per Month)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Frequency (Number of Observations)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=11)
    ax3.legend(loc='upper right', fontsize=10)
    
    # ===== PLOT 4: Geospatial distribution of total accidents by segment =====
    ax4 = axes_flat[3]
    if segment_geometry is not None:
        # Plot segment network
        if isinstance(segment_geometry, gpd.GeoDataFrame):
            segment_geometry.plot(ax=ax4, color='lightgray', linewidth=0.5, alpha=0.6)
        
        # Aggregate accidents by segment and visualize
        seg_acc = merged_panel.groupby('counter_name')[accident_col].sum().reset_index()
        seg_acc_geo = segment_geometry.merge(seg_acc, left_on='counter_name', right_on='counter_name', how='left')
        
        # Plot segments colored by accident count with colorbar
        vmin = seg_acc_geo[accident_col].min()
        vmax = seg_acc_geo[accident_col].max()
        seg_acc_geo.plot(
            ax=ax4,
            column=accident_col,
            cmap='YlOrRd',
            linewidth=1.5,
            alpha=0.8,
            legend=True,
            cax=None,
            vmin=vmin,
            vmax=vmax
        )
        ax4.set_title('Geospatial Distribution: Total Accidents by Segment', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Longitude', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Latitude', fontsize=13, fontweight='bold')
        ax4.tick_params(labelsize=10)
        ax4.ticklabel_format(style='plain', axis='both')
        ax4.grid(True, alpha=0.2)
    else:
        # Show summary statistics if no geometry
        summary_stats = merged_panel[[trip_col, accident_col]].describe()
        ax4.axis('off')
        ax4.text(0.1, 0.9, 'Summary Statistics', fontsize=14, fontweight='bold', transform=ax4.transAxes)
        
        stats_text = f"""
{trip_col}:
  Mean: {summary_stats.loc['mean', trip_col]:.0f}
  Median: {merged_panel[trip_col].median():.0f}
  Std: {summary_stats.loc['std', trip_col]:.0f}

{accident_col}:
  Mean: {summary_stats.loc['mean', accident_col]:.2f}
  Median: {merged_panel[accident_col].median():.0f}
  Std: {summary_stats.loc['std', accident_col]:.2f}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=11, transform=ax4.transAxes, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.3, wspace=0.3)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()
    
    # Print quality report after visualization
    _print_quality_checks(merged_panel, trip_col, accident_col)
    
    return fig, axes


def _print_quality_checks(
    merged_panel: pd.DataFrame,
    trip_col: str = "sum_strava_total_trip_count",
    accident_col: str = "total_accidents",
) -> None:
    """
    Internal function to print quality check report.
    Called automatically by plot_merged_panel_quality_overview.
    """
    quality_results = check_merged_panel_quality(merged_panel, trip_col, accident_col)
    
    print("\n" + "="*28 + "SUMMARY STATISTICS" + "="*28)
    for check_name, result in quality_results.items():
        print(f"\n[{result['status']}] {check_name}")
        print(f"   {result['message']}")
        print(f"   Value: {result['value']}")


def check_merged_panel_quality(
    merged_panel: pd.DataFrame,
    trip_col: str = "sum_strava_total_trip_count",
    accident_col: str = "total_accidents",
) -> dict:
    """
    Comprehensive data quality checks for merged accident-exposure panel.
    
    Returns [PASS]/[FAIL] status for each validation criterion,
    following the same pattern as raw data quality checks.
    
    Parameters
    ----------
    merged_panel : DataFrame
        Merged segment×year×month panel
    trip_col : str
        Name of trip count column
    accident_col : str
        Name of accident count column
    
    Returns
    -------
    dict
        Quality check results with status, message, and value for each check
    """
    checks = {}
    
    # 1. Temporal consistency: all segments should have all months
    temporal_check = merged_panel.groupby('counter_name').apply(
        lambda g: len(g) == merged_panel['year'].nunique() * 12
    )
    checks['temporal_consistency'] = {
        'status': 'PASS' if temporal_check.all() else 'WARN',
        'message': f'{temporal_check.sum()}/{len(temporal_check)} segments have complete temporal data',
        'value': f'{(temporal_check.sum() / len(temporal_check) * 100):.1f}%'
    }
    
    # 2. Negative values check - show which columns have negatives
    numeric_cols = merged_panel.select_dtypes(include=[np.number]).columns
    cols_with_negatives = []
    has_negatives = False
    for col in numeric_cols:
        if (merged_panel[col] < 0).any():
            has_negatives = True
            cols_with_negatives.append(col)
    
    negative_msg = 'No negative values in numeric columns' if not has_negatives else f'Found negative values in: {", ".join(cols_with_negatives)}'
    checks['no_negative_values'] = {
        'status': 'FAIL' if has_negatives else 'PASS',
        'message': negative_msg,
        'value': '0' if not has_negatives else 'Multiple'
    }
    
    # 3. Data plausibility: accidents should not exceed trips
    trip_col_name = 'sum_strava_total_trip_count'
    accident_col_name = 'total_accidents'
    
    if trip_col_name in merged_panel.columns and accident_col_name in merged_panel.columns:
        # Filter rows where both values are present
        valid_mask = merged_panel[trip_col_name].notna() & merged_panel[accident_col_name].notna()
        valid_data = merged_panel[valid_mask]
        
        if len(valid_data) > 0:
            # Count rows where accidents > trips (implausible)
            implausible = (valid_data[accident_col_name] > valid_data[trip_col_name]).sum()
            implausible_pct = (implausible / len(valid_data)) * 100
            
            plausibility_msg = f'{implausible} rows ({implausible_pct:.2f}%) have more accidents than trips'
            checks['plausibility_accidents_vs_trips'] = {
                'status': 'PASS' if implausible == 0 else 'WARN' if implausible_pct < 1 else 'FAIL',
                'message': plausibility_msg,
                'value': f'{implausible_pct:.2f}%'
            }
    
    return checks
