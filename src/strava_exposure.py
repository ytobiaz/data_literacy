from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar

from .utils import data_path


def load_strava_berlin_data(parquet_path: str | Path | None = None) -> pd.DataFrame:
    """Load Strava Berlin exposure data, downloading from Zenodo (record 15332147) if not cached locally."""
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
    """For each column, compute stability (variability within groups). 
    """
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")

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


def build_exposure_panel_segment_year_month(
    strava_berlin_data: pd.DataFrame,
    *,
    segment_static: pd.DataFrame,
    summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate Strava raw data to segment×year×month panel, applying sum/mean/mode to metrics. 
    Merges with static segment data and geometry.
    """
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
    figsize=(12, 8),
    use_tueplots=True,
    save_path=None,
):
    """Plot 2×2 grid of Strava data quality checks: monthly records, monthly trips, segment traffic distribution, daily outliers. 
    Prints validation summary.
    """
    # Apply tueplots styling if requested
    if use_tueplots:
        from tueplots import bundles
        from tueplots.constants.color import palettes
        plt.rcParams.update(bundles.icml2024(column="full", nrows=2, ncols=2))
        colors = palettes.tue_plot
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    main_color = colors[0]
    accent_color = colors[1]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Temporal coverage - Records per month (with outlier labels)
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
    
    # Total trips per month
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
    
    # Traffic distribution across segments (histogram including zeros)
    ax3 = axes[1, 0]
    if 'counter_name' in strava_df.columns and 'strava_total_trip_count' in strava_df.columns:
        df_temp = strava_df[['counter_name', 'strava_total_trip_count']].copy()
        # Include segments with missing trip counts as zero
        df_temp['strava_total_trip_count'] = df_temp['strava_total_trip_count'].fillna(0)
        
        if len(df_temp) > 0:
            # Sum trips per segment
            segment_traffic = df_temp.groupby('counter_name')['strava_total_trip_count'].sum()
            
            if len(segment_traffic) > 0:
                # Create histogram
                ax3.hist(segment_traffic.values, bins=50, color=main_color, 
                        edgecolor='black', linewidth=0.5)
                ax3.set_title('Segment Traffic Distribution', fontweight='bold', fontsize=13)
                ax3.set_xlabel('Total Trips per Segment', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Number of Segments', fontsize=14, fontweight='bold')
                ax3.tick_params(labelsize=11)
                ax3.grid(True, alpha=0.3, axis='y')
                
                # Add statistics
                zeros = (segment_traffic == 0).sum()
                total_segments = len(segment_traffic)
                zero_pct = zeros / total_segments * 100
                median = segment_traffic.median()
                mean = segment_traffic.mean()
                ax3.text(0.98, 0.97, 
                        f'Zero traffic: {zeros:,} ({zero_pct:5.1f}%)\n'
                        f'Median: {median:>8.0f}\n'
                        f'Mean:   {mean:>8.0f}', 
                        transform=ax3.transAxes, ha='right', va='top', fontsize=8,
                        family='monospace',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            else:
                ax3.text(0.5, 0.5, 'No segment data', ha='center', va='center', fontsize=11)
        else:
            ax3.text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=11)
    else:
        ax3.text(0.5, 0.5, 'counter_name or trip_count not available', ha='center', va='center', fontsize=11)
        ax3.set_title('Segment Traffic Distribution', fontweight='bold', fontsize=13)
    
    # Trip count temporal outliers
    ax4 = axes[1, 1]
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
                ax4.plot(daily_trips.index, daily_trips.values, color=main_color, 
                        linewidth=0.5, alpha=0.5)
                
                # Highlight top 5 outliers
                outliers = daily_trips[(daily_trips < lower_bound) | (daily_trips > upper_bound)].sort_values(ascending=False).head(5)
                ax4.scatter(outliers.index, outliers.values, color='red', s=30, 
                           alpha=0.7, label=f'Top 5')
                
                ax4.set_title('Daily Trip Count', fontweight='bold', fontsize=13)
                ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
                ax4.set_ylabel('Total Daily Trips', fontsize=14, fontweight='bold')
                ax4.tick_params(labelsize=9, axis='x', rotation=45)
                ax4.tick_params(labelsize=11, axis='y')
                ax4.grid(True, alpha=0.3, axis='y')
                ax4.legend(loc='upper left', fontsize=8)
            else:
                ax4.text(0.5, 0.5, 'No daily data', ha='center', va='center', fontsize=11)
        else:
            ax4.text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=11)
    else:
        ax4.text(0.5, 0.5, 'date or trip_count not available', ha='center', va='center', fontsize=11)
        ax4.set_title('Daily Trip Count Outliers', fontweight='bold', fontsize=13)
    
    # Adjust layout
    fig.subplots_adjust(hspace=0.4, wspace=0.35)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    # Store validation results for summary (not plotted)
    validation_results = {}
    
    # Date range check (2019-2023)
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
    
    
    # No duplicate records (counter_name + date)
    if 'counter_name' in strava_df.columns and 'date' in strava_df.columns:
        duplicates = strava_df.duplicated(subset=['counter_name', 'date'], keep=False).sum()
        validation_results['No Duplicate Records (segment+date)'] = (
            duplicates == 0,
            f'Unique records by segment+date' if duplicates == 0 else f'Duplicate records: {duplicates:,}'
        )
    
    # Non-negative trip counts
    if 'strava_total_trip_count' in strava_df.columns:
        trip_counts = strava_df['strava_total_trip_count'].dropna()
        if len(trip_counts) > 0:
            negative_counts = (trip_counts < 0).sum()
            neg_cols_with_negs = []
            if negative_counts > 0:
                neg_cols_with_negs = ['strava_total_trip_count']
            validation_results['Non-negative Trip Counts'] = (
                negative_counts == 0,
                f'All trip counts non-negative' if negative_counts == 0 else f'Negative trip counts: {negative_counts:,} in columns: {", ".join(neg_cols_with_negs)}'
            )
    
    # Segments with zero total traffic (should have no zero-traffic segments)
    if 'counter_name' in strava_df.columns and 'strava_total_trip_count' in strava_df.columns:
        df_temp = strava_df[['counter_name', 'strava_total_trip_count']].copy()
        df_temp['strava_total_trip_count'] = df_temp['strava_total_trip_count'].fillna(0)
        segment_traffic = df_temp.groupby('counter_name')['strava_total_trip_count'].sum()
        zero_segments = (segment_traffic == 0).sum()
        zero_segment_pct = zero_segments / len(segment_traffic) * 100
        # Should have zero segments with zero traffic
        zero_traffic_ok = zero_segments == 0
        validation_results['No Segments with Zero Traffic'] = (
            zero_traffic_ok,
            f'Zero traffic found in {zero_segments} segments ({zero_segment_pct:.1f}%)' if zero_segments > 0 else 'No segments with zero traffic'
        )
    
    # Segment coverage consistency
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
    
    # Temporal coverage and gaps check
    if 'date' in strava_df.columns:
        dates = pd.to_datetime(strava_df['date'], errors='coerce')
        valid_dates = dates.dropna()
        
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            date_range = pd.date_range(min_date, max_date, freq='D')
            daily_counts = valid_dates.value_counts()
            gaps = [d for d in date_range if d not in daily_counts.index]
            gap_pct = len(gaps) / len(date_range) * 100
            # Less than 5% missing days is acceptable
            gaps_ok = gap_pct < 5
            validation_results['Temporal Coverage'] = (
                gaps_ok,
                f'{min_date.date()} to {max_date.date()}, {len(gaps)} missing days ({gap_pct:.1f}% of {len(date_range)})'
            )
    
    # Missing values check in coordinates
    if 'latitude' in strava_df.columns and 'longitude' in strava_df.columns:
        lat_missing = strava_df['latitude'].isna().sum()
        lon_missing = strava_df['longitude'].isna().sum()
        coord_missing_ok = (lat_missing == 0) and (lon_missing == 0)
        coord_desc = 'All coordinates present' if coord_missing_ok else f'Missing: latitude={lat_missing}, longitude={lon_missing}'
        validation_results['Missing Values in Coordinates'] = (
            coord_missing_ok,
            coord_desc
        )
    
    # Print summary statistics
    print("\n" + "="*21 + " SUMMARY STATISTICS " + "="*21)
    print(f"Total records: {len(strava_df):,}")
    print(f"Number of columns: {len(strava_df.columns)}")
    
    # Data Quality Checks
    print("\nData Quality Checks:")
    for test_name, (result, description) in validation_results.items():
        status = '[PASS]' if result else '[FAIL]'
        print(f"{status} {test_name}")
        print(f"  {description}")
    
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
    


def get_daily_outlier_days(strava_df):
    """Identify days with outlier trip counts using IQR method."""
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


def filter_zero_traffic_years(exposure_panel: pd.DataFrame, trip_col: str = 'sum_strava_total_trip_count') -> pd.DataFrame:
    """Remove segments with any year having zero traffic. Ensures temporal consistency across segment-year observations."""
    # Identify segments with zero traffic in ANY year
    seg_year_trips = exposure_panel.groupby(['counter_name', 'year'])[trip_col].sum()
    zero_year_segs = seg_year_trips[seg_year_trips == 0].index.get_level_values('counter_name').unique()
    
    # Filter to keep only segments without zero-traffic years
    filtered = exposure_panel[~exposure_panel['counter_name'].isin(zero_year_segs)].copy()
    
    n_removed = exposure_panel['counter_name'].nunique() - filtered['counter_name'].nunique()
    print(f"Removed {n_removed} segments with zero traffic in any year")
    print(f"Segments remaining: {filtered['counter_name'].nunique():,} (from {exposure_panel['counter_name'].nunique():,})")
    
    return filtered


def compute_segment_filter_impact(exposure_panel: pd.DataFrame, trip_col: str = 'sum_strava_total_trip_count') -> pd.DataFrame:
    """Quantify segment loss from 3 filtering criteria: zero-traffic all-time, any-year, or any-month."""
    if 'counter_name' not in exposure_panel.columns:
        print("ERROR: 'counter_name' column not found in exposure panel")
        return pd.DataFrame()
    
    # Find trip count column
    if trip_col not in exposure_panel.columns:
        trip_col = 'sum_strava_total_trip_count'
        if trip_col not in exposure_panel.columns:
            print(f"ERROR: No trip count column found. Available columns: {exposure_panel.columns.tolist()}")
            return pd.DataFrame()
    
    # Ensure trip_col is numeric
    exposure_panel = exposure_panel.copy()
    exposure_panel[trip_col] = pd.to_numeric(exposure_panel[trip_col], errors='coerce').fillna(0)
    
    baseline_segments = exposure_panel['counter_name'].nunique()
    results = []
    
    # Criterion 1: Segments with zero total traffic across all time
    seg_total_trips = exposure_panel.groupby('counter_name')[trip_col].sum()
    segments_nonzero = (seg_total_trips > 0).sum()
    loss_nonzero = baseline_segments - segments_nonzero
    loss_pct_nonzero = loss_nonzero / baseline_segments * 100
    
    results.append({
        'Filter Criterion': 'Remove segments with 0 traffic (all time)',
        'Segments Remaining': segments_nonzero,
        'Segments Lost': loss_nonzero,
        'Loss %': loss_pct_nonzero
    })
    
    # Criterion 2: Segments with zero traffic in ANY year
    seg_year_trips = exposure_panel.groupby(['counter_name', 'year'])[trip_col].sum()
    zero_year_segs = seg_year_trips[seg_year_trips == 0].index.get_level_values('counter_name').unique()
    segments_no_zero_years = baseline_segments - len(zero_year_segs)
    loss_zero_years = len(zero_year_segs)
    loss_pct_zero_years = loss_zero_years / baseline_segments * 100
    
    results.append({
        'Filter Criterion': 'Remove segments with 0 traffic in any year',
        'Segments Remaining': segments_no_zero_years,
        'Segments Lost': loss_zero_years,
        'Loss %': loss_pct_zero_years
    })
    
    # Criterion 3: Segments with zero traffic in ANY month
    zero_month_segs = exposure_panel[exposure_panel[trip_col] == 0]['counter_name'].unique()
    segments_no_zero_months = baseline_segments - len(zero_month_segs)
    loss_zero_months = len(zero_month_segs)
    loss_pct_zero_months = loss_zero_months / baseline_segments * 100
    
    results.append({
        'Filter Criterion': 'Remove segments with 0 traffic in any month',
        'Segments Remaining': segments_no_zero_months,
        'Segments Lost': loss_zero_months,
        'Loss %': loss_pct_zero_months
    })
    
    # Format results
    results_df = pd.DataFrame(results)
    
    return results_df



