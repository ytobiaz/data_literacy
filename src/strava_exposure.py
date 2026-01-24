from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar

from .utils import data_path


def load_strava_berlin_data(parquet_path: str | Path | None = None) -> pd.DataFrame:
    parquet_path = (
        Path(parquet_path) if parquet_path is not None else data_path("strava", "berlin_data.parquet")
    )
    if not parquet_path.exists():
        # If we don't have a local copy, download from Zenodo (original source).
        base_url = "https://zenodo.org/records/15332147/files"
        url = f"{base_url}/berlin_data.parquet?download=1"
        print(f"Downloading berlin_data.parquet from Zenodo...")
        df = pd.read_parquet(url, engine="pyarrow")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path)
        print(f"Saved berlin_data.parquet to {parquet_path}")
        return df
    print(f"Reading berlin_data.parquet locally from {parquet_path}")
    return pd.read_parquet(parquet_path)


def column_stability_summary(df: pd.DataFrame, *, group_col: str = "counter_name") -> pd.DataFrame:
    """For each column, compute whether it varies within a segment over time.
    
    Optimized version using vectorized operations instead of iterating over columns.

    Returns a dataframe with columns: column, segments_total, segments_varying, max_unique_within_any_segment.
    """

    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    # Get all columns except group_col
    cols_to_check = [col for col in df.columns if col != group_col]
    
    # Vectorized computation: compute nunique for all columns at once
    nunique_all = df.groupby(group_col, sort=False)[cols_to_check].nunique()
    
    # Build summary dataframe directly
    summary_df = pd.DataFrame({
        "column": cols_to_check,
        "segments_total": len(nunique_all),
        "segments_varying": [(nunique_all[col] > 1).sum() for col in cols_to_check],
        "max_unique_within_any_segment": [nunique_all[col].max() for col in cols_to_check],
    })

    return summary_df.sort_values("segments_varying", ascending=True).reset_index(drop=True)


def categorize_strava_features(summary_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Categorize Strava columns into feature groups and compute statistics.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from column_stability_summary with columns: column, segments_total, segments_varying
    df : pd.DataFrame
        Original Strava dataframe to extract dtypes
        
    Returns
    -------
    pd.DataFrame
        Summary statistics by feature category with columns: total_cols, constant_cols, varying_cols, percent_constant
    """
    
    def _categorize_column(col_name: str) -> str:
        """Assign column to a feature category based on name."""
        col_lower = col_name.lower()
        if any(x in col_lower for x in ['street', 'lane', 'speed', 'maxspeed', 'road', 'surface', 'width']):
            return 'Infrastructure'
        elif any(x in col_lower for x in ['degree', 'betweenness', 'closeness']):
            return 'Connectivity'
        elif any(x in col_lower for x in ['strava', 'count', 'trip']):
            return 'Strava'
        elif any(x in col_lower for x in ['weather', 'temp', 'rain', 'sunshine', 'wind']):
            return 'Weather'
        elif any(x in col_lower for x in ['motorized', 'car', 'vehicle']):
            return 'Motorized'
        elif any(x in col_lower for x in ['population', 'income', 'unemployment']):
            return 'Socioeconomic'
        else:
            return 'Other'
    
    # Add category column
    summary_with_cat = summary_df.copy()
    summary_with_cat['category'] = summary_with_cat['column'].apply(_categorize_column)
    
    # Aggregate by category
    category_stats = summary_with_cat.groupby('category').agg(
        total_cols=('column', 'count'),
        constant_cols=('segments_varying', lambda s: (s == 0).sum()),
        varying_cols=('segments_varying', lambda s: (s > 0).sum()),
    )
    
    category_stats['percent_constant'] = (
        category_stats['constant_cols'] / category_stats['total_cols'] * 100
    ).round(1)
    
    return category_stats.sort_values('percent_constant', ascending=False)


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


def plot_strava_quality_overview(
    strava_df,
    figsize=(16, 8.4),
    use_tueplots=True,
    save_path=None,
):
    """
    Create comprehensive quality check visualization for Strava exposure data.
    
    Parameters
    ----------
    strava_df : DataFrame
        Strava Berlin data with counter_name, date, strava_total_trip_count, etc.
    figsize : tuple, optional
        Figure size (width, height), by default (16, 8.4)
    use_tueplots : bool, optional
        Whether to use tueplots ICML2024 stylesheet, by default True
    save_path : str | Path | None, optional
        Path to save the figure, by default None
    """
    # Apply tueplots styling if requested
    if use_tueplots:
        from tueplots import bundles
        from tueplots.constants.color import palettes
        plt.rcParams.update(bundles.icml2024(column="full", nrows=2, ncols=3))
        colors = palettes.tue_plot
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    main_color = colors[0]
    accent_color = colors[1]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Temporal coverage - Records per month (with outlier labels)
    ax1 = axes[0, 0]
    if 'date' in strava_df.columns:
        dates = pd.to_datetime(strava_df['date'], errors='coerce')
        year_month = dates.dt.to_period('M').dropna()
        if len(year_month) > 0:
            counts = year_month.value_counts().sort_index()
            
            # Plot line
            ax1.plot(counts.index.astype(str), counts.values, color=main_color, linewidth=2, marker='o', markersize=3)
            ax1.set_title('Records per Month', fontweight='bold', fontsize=13)
            ax1.set_xlabel('Year-Month', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Record Count', fontsize=14, fontweight='bold')
            ax1.tick_params(labelsize=9, axis='x', rotation=45)
            ax1.tick_params(labelsize=11, axis='y')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits (min - 10000, max + 10000)
            y_min = max(0, counts.min() - 10000)
            y_max = counts.max() + 10000
            ax1.set_ylim(y_min, y_max)
            
            # Label outliers (only low values - bottom 3)
            sorted_counts = counts.sort_values()
            outliers_low = sorted_counts.head(3)
            
            for period, value in outliers_low.items():
                period_str = str(period)
                x_pos = list(counts.index).index(period)
                ax1.annotate(period_str, xy=(x_pos, value), 
                           xytext=(0, -15),
                           textcoords='offset points', ha='center', 
                           fontsize=7, color='red', fontweight='bold')
            
            # Show every 6th label
            xticks = ax1.get_xticks()
            if len(xticks) > 12:
                ax1.set_xticks(xticks[::6])
        else:
            ax1.text(0.5, 0.5, 'No valid dates', ha='center', va='center', fontsize=11)
    else:
        ax1.text(0.5, 0.5, 'date column not available', ha='center', va='center', fontsize=11)
        ax1.set_title('Records per Month', fontweight='bold', fontsize=13)
    
    # 2. Total trips per month
    ax2 = axes[0, 1]
    if 'date' in strava_df.columns and 'strava_total_trip_count' in strava_df.columns:
        dates = pd.to_datetime(strava_df['date'], errors='coerce')
        df_temp = pd.DataFrame({'date': dates, 'trips': strava_df['strava_total_trip_count']})
        df_temp = df_temp.dropna(subset=['date', 'trips'])
        
        if len(df_temp) > 0:
            df_temp['year_month'] = df_temp['date'].dt.to_period('M')
            monthly_trips = df_temp.groupby('year_month')['trips'].sum().sort_index()
            
            ax2.plot(monthly_trips.index.astype(str), monthly_trips.values, 
                    color=main_color, linewidth=2, marker='o', markersize=3)
            ax2.set_title('Total Trips per Month', fontweight='bold', fontsize=13)
            ax2.set_xlabel('Year-Month', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Total Strava Trips', fontsize=14, fontweight='bold')
            ax2.tick_params(labelsize=9, axis='x', rotation=45)
            ax2.tick_params(labelsize=11, axis='y')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Show every 6th label
            xticks = ax2.get_xticks()
            if len(xticks) > 12:
                ax2.set_xticks(xticks[::6])
        else:
            ax2.text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=11)
    else:
        ax2.text(0.5, 0.5, 'date or strava_total_trip_count not available', ha='center', va='center', fontsize=11)
        ax2.set_title('Total Trips per Month', fontweight='bold', fontsize=13)
    
    # 3. Segment coverage over time
    ax3 = axes[0, 2]
    if 'date' in strava_df.columns and 'counter_name' in strava_df.columns:
        dates = pd.to_datetime(strava_df['date'], errors='coerce')
        df_temp = pd.DataFrame({'date': dates, 'counter_name': strava_df['counter_name']})
        df_temp = df_temp.dropna(subset=['date'])
        df_temp['year_month'] = df_temp['date'].dt.to_period('M')
        
        if len(df_temp) > 0:
            segment_counts = df_temp.groupby('year_month')['counter_name'].nunique()
            ax3.plot(segment_counts.index.astype(str), segment_counts.values, 
                    color=main_color, linewidth=2, marker='o', markersize=3)
            ax3.set_title('Unique Segments per Month', fontweight='bold', fontsize=13)
            ax3.set_xlabel('Year-Month', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Unique Segments', fontsize=14, fontweight='bold')
            ax3.tick_params(labelsize=9, axis='x', rotation=45)
            ax3.tick_params(labelsize=11, axis='y')
            ax3.grid(True, alpha=0.3, axis='y')
            # Show every 6th label
            xticks = ax3.get_xticks()
            if len(xticks) > 12:
                ax3.set_xticks(xticks[::6])
        else:
            ax3.text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=11)
    else:
        ax3.text(0.5, 0.5, 'date or counter_name not available', ha='center', va='center', fontsize=11)
        ax3.set_title('Segment Coverage', fontweight='bold', fontsize=13)
    
    # 4. Traffic distribution across segments (histogram including zeros)
    ax4 = axes[1, 0]
    if 'counter_name' in strava_df.columns and 'strava_total_trip_count' in strava_df.columns:
        df_temp = strava_df[['counter_name', 'strava_total_trip_count']].copy()
        # Include segments with missing trip counts as zero
        df_temp['strava_total_trip_count'] = df_temp['strava_total_trip_count'].fillna(0)
        
        if len(df_temp) > 0:
            # Sum trips per segment
            segment_traffic = df_temp.groupby('counter_name')['strava_total_trip_count'].sum()
            
            if len(segment_traffic) > 0:
                # Create histogram
                ax4.hist(segment_traffic.values, bins=50, color=main_color, 
                        edgecolor='black', linewidth=0.5)
                ax4.set_title('Segment Traffic Distribution', fontweight='bold', fontsize=13)
                ax4.set_xlabel('Total Trips per Segment', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Number of Segments', fontsize=14, fontweight='bold')
                ax4.tick_params(labelsize=11)
                ax4.grid(True, alpha=0.3, axis='y')
                
                # Add statistics
                zeros = (segment_traffic == 0).sum()
                total_segments = len(segment_traffic)
                zero_pct = zeros / total_segments * 100
                median = segment_traffic.median()
                mean = segment_traffic.mean()
                ax4.text(0.95, 0.95, 
                        f'Zero traffic: {zeros}/{total_segments} ({zero_pct:.1f}%)\n'
                        f'Median: {median:.0f}\n'
                        f'Mean: {mean:.0f}', 
                        transform=ax4.transAxes, ha='right', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax4.text(0.5, 0.5, 'No segment data', ha='center', va='center', fontsize=11)
        else:
            ax4.text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=11)
    else:
        ax4.text(0.5, 0.5, 'counter_name or trip_count not available', ha='center', va='center', fontsize=11)
        ax4.set_title('Segment Traffic Distribution', fontweight='bold', fontsize=13)
    
    # 5. Segment-level outliers (box plot)
    ax5 = axes[1, 1]
    if 'counter_name' in strava_df.columns and 'strava_total_trip_count' in strava_df.columns:
        df_temp = strava_df[['counter_name', 'strava_total_trip_count']].copy()
        df_temp['strava_total_trip_count'] = df_temp['strava_total_trip_count'].fillna(0)
        
        if len(df_temp) > 0:
            segment_traffic = df_temp.groupby('counter_name')['strava_total_trip_count'].sum()
            
            if len(segment_traffic) > 0:
                bp = ax5.boxplot([segment_traffic.values], vert=False, widths=0.5,
                                 patch_artist=True,
                                 boxprops=dict(facecolor=main_color, alpha=0.7),
                                 medianprops=dict(color='red', linewidth=2),
                                 whiskerprops=dict(color=main_color, linewidth=1.5),
                                 capprops=dict(color=main_color, linewidth=1.5))
                ax5.set_title('Segment Traffic Outliers', fontweight='bold', fontsize=13)
                ax5.set_xlabel('Total Trips per Segment', fontsize=14, fontweight='bold')
                ax5.set_yticks([])
                ax5.tick_params(labelsize=11)
                ax5.grid(True, alpha=0.3, axis='x')
                
                # Add outlier count
                Q1 = segment_traffic.quantile(0.25)
                Q3 = segment_traffic.quantile(0.75)
                IQR = Q3 - Q1
                outliers = segment_traffic[(segment_traffic < Q1 - 1.5*IQR) | (segment_traffic > Q3 + 1.5*IQR)]
                ax5.text(0.95, 0.95, f'Outliers: {len(outliers)}/{len(segment_traffic)}', 
                        transform=ax5.transAxes, ha='right', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax5.text(0.5, 0.5, 'No segment data', ha='center', va='center', fontsize=11)
        else:
            ax5.text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=11)
    else:
        ax5.text(0.5, 0.5, 'Required columns not available', ha='center', va='center', fontsize=11)
        ax5.set_title('Segment Traffic Outliers', fontweight='bold', fontsize=13)
    
    # 6. Trip count temporal outliers
    ax6 = axes[1, 2]
    if 'date' in strava_df.columns and 'strava_total_trip_count' in strava_df.columns:
        dates = pd.to_datetime(strava_df['date'], errors='coerce')
        df_temp = pd.DataFrame({'date': dates, 'trips': strava_df['strava_total_trip_count']})
        df_temp = df_temp.dropna(subset=['date', 'trips'])
        
        if len(df_temp) > 0:
            # Aggregate by day
            daily_trips = df_temp.groupby('date')['trips'].sum().sort_index()
            
            if len(daily_trips) > 0:
                # Calculate outliers
                Q1 = daily_trips.quantile(0.25)
                Q3 = daily_trips.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Plot
                ax6.plot(daily_trips.index, daily_trips.values, color=main_color, 
                        linewidth=0.5, alpha=0.5)
                
                # Highlight outliers
                outliers = daily_trips[(daily_trips < lower_bound) | (daily_trips > upper_bound)]
                ax6.scatter(outliers.index, outliers.values, color='red', s=20, 
                           alpha=0.7, label=f'Outliers ({len(outliers)})')
                
                ax6.axhline(y=upper_bound, color='red', linestyle='--', linewidth=1, alpha=0.5)
                ax6.axhline(y=lower_bound, color='red', linestyle='--', linewidth=1, alpha=0.5)
                
                ax6.set_title('Daily Trip Count Outliers', fontweight='bold', fontsize=13)
                ax6.set_xlabel('Date', fontsize=11, fontweight='bold')
                ax6.set_ylabel('Total Daily Trips', fontsize=14, fontweight='bold')
                ax6.tick_params(labelsize=9, axis='x', rotation=45)
                ax6.tick_params(labelsize=11, axis='y')
                ax6.grid(True, alpha=0.3, axis='y')
                ax6.legend(loc='upper right', fontsize=8)
            else:
                ax6.text(0.5, 0.5, 'No daily data', ha='center', va='center', fontsize=11)
        else:
            ax6.text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=11)
    else:
        ax6.text(0.5, 0.5, 'date or trip_count not available', ha='center', va='center', fontsize=11)
        ax6.set_title('Daily Trip Count Outliers', fontweight='bold', fontsize=13)
    
    # Adjust layout
    fig.subplots_adjust(hspace=0.4, wspace=0.35)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    # Store validation results for summary (not plotted)
    validation_results = {}
    
    # Test 1: Date range check (2019-2023)
    if 'date' in strava_df.columns:
        dates = pd.to_datetime(strava_df['date'], errors='coerce')
        valid_dates = dates.dropna()
        if len(valid_dates) > 0:
            min_year = valid_dates.dt.year.min()
            max_year = valid_dates.dt.year.max()
            date_range_ok = (min_year >= 2019) and (max_year <= 2023)
            validation_results['Temporal Coverage (2019-2023)'] = (
                date_range_ok, 
                f'Date range: {min_year}-{max_year} (expected: 2019-2023)'
            )
        else:
            validation_results['Temporal Coverage (2019-2023)'] = (
                False, 
                'No valid dates found'
            )
    
    # Test 2: Required columns present
    required_cols = ['counter_name', 'date', 'strava_total_trip_count', 'latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in strava_df.columns]
    validation_results['Required Columns Present'] = (
        len(missing_cols) == 0,
        f'All required columns present' if len(missing_cols) == 0 else f'Missing columns: {missing_cols}'
    )
    
    # Test 3: No duplicate records (counter_name + date)
    if 'counter_name' in strava_df.columns and 'date' in strava_df.columns:
        duplicates = strava_df.duplicated(subset=['counter_name', 'date'], keep=False).sum()
        validation_results['No Duplicate Records (segment+date)'] = (
            duplicates == 0,
            f'Unique records by segment+date' if duplicates == 0 else f'Duplicate records: {duplicates:,}'
        )
    
    # Test 4: Positive trip counts
    if 'strava_total_trip_count' in strava_df.columns:
        trip_counts = strava_df['strava_total_trip_count'].dropna()
        if len(trip_counts) > 0:
            negative_counts = (trip_counts < 0).sum()
            validation_results['Non-negative Trip Counts'] = (
                negative_counts == 0,
                f'All trip counts non-negative' if negative_counts == 0 else f'Negative trip counts: {negative_counts:,}'
            )
    
    # Test 5: Segment coverage consistency
    if 'counter_name' in strava_df.columns:
        n_unique_segments = strava_df['counter_name'].nunique()
        total_records = len(strava_df)
        avg_records_per_segment = total_records / n_unique_segments if n_unique_segments > 0 else 0
        # Expecting roughly daily records over ~5 years = ~1825 records per segment
        coverage_ok = avg_records_per_segment >= 1000  # At least 1000 records per segment on average
        validation_results['Segment Coverage Consistency'] = (
            coverage_ok,
            f'{n_unique_segments:,} unique segments, avg {avg_records_per_segment:.0f} records/segment'
        )
    
    # Test 6: Segment-level outliers check
    if 'counter_name' in strava_df.columns and 'strava_total_trip_count' in strava_df.columns:
        df_temp = strava_df[['counter_name', 'strava_total_trip_count']].copy()
        df_temp['strava_total_trip_count'] = df_temp['strava_total_trip_count'].fillna(0)
        segment_traffic = df_temp.groupby('counter_name')['strava_total_trip_count'].sum()
        
        if len(segment_traffic) > 0:
            Q1 = segment_traffic.quantile(0.25)
            Q3 = segment_traffic.quantile(0.75)
            IQR = Q3 - Q1
            outliers = segment_traffic[(segment_traffic < Q1 - 1.5*IQR) | (segment_traffic > Q3 + 1.5*IQR)]
            outlier_pct = len(outliers) / len(segment_traffic) * 100
            # Outliers should be less than 10% of segments
            outliers_ok = outlier_pct < 10
            validation_results['Segment Traffic Outliers (<10%)'] = (
                outliers_ok,
                f'{len(outliers)} outliers ({outlier_pct:.1f}% of {len(segment_traffic)} segments)'
            )
    
    # Test 7: Temporal gaps check
    if 'date' in strava_df.columns:
        dates = pd.to_datetime(strava_df['date'], errors='coerce')
        valid_dates = dates.dropna()
        
        if len(valid_dates) > 0:
            date_range = pd.date_range(valid_dates.min(), valid_dates.max(), freq='D')
            daily_counts = valid_dates.value_counts()
            gaps = [d for d in date_range if d not in daily_counts.index]
            gap_pct = len(gaps) / len(date_range) * 100
            # Less than 5% missing days is acceptable
            gaps_ok = gap_pct < 5
            validation_results['Temporal Coverage Gaps (<5%)'] = (
                gaps_ok,
                f'{len(gaps)} missing days ({gap_pct:.1f}% of {len(date_range)} days)'
            )
    
    # Test 8: Missing data in key columns
    key_columns = ['date', 'counter_name', 'strava_total_trip_count']
    available_cols = [col for col in key_columns if col in strava_df.columns]
    
    if len(available_cols) > 0:
        max_missing_pct = 0
        max_missing_col = None
        for col in available_cols:
            missing_pct = strava_df[col].isna().sum() / len(strava_df) * 100
            if missing_pct > max_missing_pct:
                max_missing_pct = missing_pct
                max_missing_col = col
        
        # Key columns should have less than 10% missing
        missing_ok = max_missing_pct < 10
        validation_results['Key Columns Missing Data (<10%)'] = (
            missing_ok,
            f'Max missing: {max_missing_pct:.1f}% in {max_missing_col}' if max_missing_col else 'No missing data'
        )
    
    # Test 9: Daily trip count outliers
    if 'date' in strava_df.columns and 'strava_total_trip_count' in strava_df.columns:
        dates = pd.to_datetime(strava_df['date'], errors='coerce')
        df_temp = pd.DataFrame({'date': dates, 'trips': strava_df['strava_total_trip_count']})
        df_temp = df_temp.dropna(subset=['date', 'trips'])
        
        if len(df_temp) > 0:
            daily_trips = df_temp.groupby('date')['trips'].sum()
            
            if len(daily_trips) > 0:
                Q1 = daily_trips.quantile(0.25)
                Q3 = daily_trips.quantile(0.75)
                IQR = Q3 - Q1
                outliers = daily_trips[(daily_trips < Q1 - 1.5*IQR) | (daily_trips > Q3 + 1.5*IQR)]
                outlier_pct = len(outliers) / len(daily_trips) * 100
                # Less than 5% daily outliers is acceptable
                outliers_ok = outlier_pct < 5
                validation_results['Daily Trip Count Outliers (<5%)'] = (
                    outliers_ok,
                    f'{len(outliers)} outlier days ({outlier_pct:.1f}% of {len(daily_trips)} days)'
                )
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS - STRAVA EXPOSURE DATA")
    print("="*70)
    print(f"Total records: {len(strava_df):,}")
    print(f"Number of columns: {len(strava_df.columns)}")
    
    # Data Quality Checks
    print("\n" + "-"*70)
    print("DATA QUALITY CHECKS")
    print("-"*70)
    for test_name, (result, description) in validation_results.items():
        status = 'PASS' if result else 'FAIL'
        print(f"\n[{status}] {test_name}")
        print(f"  Description: {description}")
    
    # Unique segments
    if 'counter_name' in strava_df.columns:
        print(f"\nUnique segments: {strava_df['counter_name'].nunique():,}")
        
        # Segment traffic distribution
        if 'strava_total_trip_count' in strava_df.columns:
            df_temp = strava_df[['counter_name', 'strava_total_trip_count']].copy()
            df_temp['strava_total_trip_count'] = df_temp['strava_total_trip_count'].fillna(0)
            segment_traffic = df_temp.groupby('counter_name')['strava_total_trip_count'].sum()
            
            if len(segment_traffic) > 0:
                zeros = (segment_traffic == 0).sum()
                print(f"  - Segments with zero trips: {zeros}")
                print(f"  - Segment traffic median: {segment_traffic.median():.0f}")
                print(f"  - Segment traffic max: {segment_traffic.max():.0f}")
    
    # Temporal coverage
    if 'date' in strava_df.columns:
        dates = pd.to_datetime(strava_df['date'], errors='coerce')
        valid_dates = dates.dropna()
        
        if len(valid_dates) > 0:
            date_range = pd.date_range(valid_dates.min(), valid_dates.max(), freq='D')
            daily_counts = valid_dates.value_counts()
            gaps = [d for d in date_range if d not in daily_counts.index]
            
            print(f"\nTemporal coverage:")
            print(f"  - Total days in range: {len(date_range)}")
            print(f"  - Days with data: {len(daily_counts)}")
            print(f"  - Missing days (gaps): {len(gaps)} ({len(gaps)/len(date_range)*100:.1f}%)")
    
    # Trip count statistics
    if 'strava_total_trip_count' in strava_df.columns:
        trip_counts = strava_df['strava_total_trip_count'].dropna()
        if len(trip_counts) > 0:
            print(f"\nTrip count statistics:")
            print(f"  - Mean: {trip_counts.mean():.1f}")
            print(f"  - Median: {trip_counts.median():.1f}")
            print(f"  - Min: {trip_counts.min():.1f}")
            print(f"  - Max: {trip_counts.max():.1f}")
            print(f"  - Total trips: {trip_counts.sum():,.0f}")
    
    # Missing values
    missing_by_col = strava_df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    
    if len(missing_by_col) > 0:
        print(f"\nMissing values by column (top 10):")
        for col, count in list(missing_by_col.items())[:10]:
            pct = count / len(strava_df) * 100
            print(f"  {col}: {count:,} ({pct:.1f}%)")
    else:
        print(f"\nNo missing values detected!")
    
    print("="*70)


def get_daily_outlier_days(strava_df):
    """
    Extract and display days with suspicious (outlier) trip counts.
    
    Parameters
    ----------
    strava_df : DataFrame
        Strava data with 'date' and 'strava_total_trip_count' columns
        
    Returns
    -------
    DataFrame
        Days with outlier trip counts sorted by trip count descending
    """
    if 'date' not in strava_df.columns or 'strava_total_trip_count' not in strava_df.columns:
        print("Required columns 'date' and 'strava_total_trip_count' not found")
        return pd.DataFrame()
    
    dates = pd.to_datetime(strava_df['date'], errors='coerce')
    df_temp = pd.DataFrame({'date': dates, 'trips': strava_df['strava_total_trip_count']})
    df_temp = df_temp.dropna(subset=['date', 'trips'])
    
    if len(df_temp) == 0:
        print("No valid data found")
        return pd.DataFrame()
    
    # Aggregate by day
    daily_trips = df_temp.groupby('date')['trips'].sum().sort_index()
    
    if len(daily_trips) == 0:
        print("No daily data found")
        return pd.DataFrame()
    
    # Calculate outliers using IQR method
    Q1 = daily_trips.quantile(0.25)
    Q3 = daily_trips.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Get outliers (upper bound = suspicious high values)
    outlier_mask = (daily_trips < lower_bound) | (daily_trips > upper_bound)
    outlier_days = daily_trips[outlier_mask].sort_values(ascending=False)
    
    if len(outlier_days) == 0:
        print("No outlier days found")
        return pd.DataFrame()
    
    # Format results
    result = pd.DataFrame({
        'Date': outlier_days.index.strftime('%Y-%m-%d'),
        'Day of Week': outlier_days.index.strftime('%A'),
        'Trip Count': outlier_days.values.astype(int),
        'Type': ['High' if x > upper_bound else 'Low' for x in outlier_days.values]
    }).reset_index(drop=True)
    
    return result

