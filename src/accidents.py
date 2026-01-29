from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import geopandas as gpd
from geopandas import GeoDataFrame

from tueplots import bundles
from tueplots.constants.color import palettes

from .utils import data_path



def assign_accidents_to_nearest_segment(
    accidents_bike_berlin: pd.DataFrame,
    segments_gdf: GeoDataFrame,
    *,
    canonical_crs: str = "EPSG:32633",
    max_distance_m: float = 10.0,
    x_col: str = "XGCSWGS84",
    y_col: str = "YGCSWGS84",
) -> GeoDataFrame:
    """Assign each accident to exactly one nearest segment (within max_distance_m)."""

    accidents = accidents_bike_berlin.reset_index(drop=True).copy()

    if x_col not in accidents.columns or y_col not in accidents.columns:
        raise KeyError(f"Expected coordinate columns {x_col!r}, {y_col!r} in accidents dataframe")

    accident_locations_gdf = gpd.GeoDataFrame(
        accidents,
        geometry=gpd.points_from_xy(accidents[x_col], accidents[y_col]),
        crs="EPSG:4326",
    ).to_crs(canonical_crs)

    accident_locations_gdf = accident_locations_gdf.reset_index(drop=True)
    accident_locations_gdf["acc_id"] = accident_locations_gdf.index

    joined = gpd.sjoin_nearest(
        accident_locations_gdf,
        segments_gdf,
        how="left",
        max_distance=max_distance_m,
        distance_col="dist",
    )

    joined = joined.dropna(subset=["index_right"]).copy()

    joined_nearest_unique = (
        joined.sort_values("dist").drop_duplicates(subset=["acc_id"], keep="first")
    )

    return joined_nearest_unique

# TODO: delete this function if not needed
def add_temporal_features(accidents_bike_berlin: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features (weekday_type and time_of_day) to accident dataframe.
    
    weekday_type: 'weekday' (Mon-Fri) vs 'weekend' (Sat-Sun)
        Note: weekday column encoding: 1=Sunday, 2=Monday, ..., 7=Saturday
    
    time_of_day: 'work_hours (7h-18h)', 'evening (18h-22h)', 'night (22h-7h)'
    
    Parameters
    ----------
    accidents_bike_berlin : pd.DataFrame
        Accident dataframe with 'weekday' and 'hour' columns
        
    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'weekday_type' and 'time_of_day' columns
    """
    accidents = accidents_bike_berlin.copy()
    
    # weekday_type: weekday (Mon-Fri) vs weekend (Sat-Sun)
    # Day of the week: 1=Sunday, 2=Monday, ..., 7=Saturday
    if "weekday" in accidents.columns:
        accidents["weekday_type"] = accidents["weekday"].map(
            lambda x: "weekday" if x in [2, 3, 4, 5, 6] else "weekend"
        )
    else:
        print("Warning: 'weekday' column not found, skipping weekday_type")

    # time_of_day: work_hours (7-18), evening (18-22), night (22-7)
    if "hour" in accidents.columns:
        def _classify_time_of_day(hour):
            if pd.isna(hour):
                return None
            h = int(hour)
            if 7 <= h < 18:
                return "work_hours (7h-18h)"
            elif 18 <= h < 22:
                return "evening (18h-22h)"
            else:
                return "night (22h-7h)"
        
        accidents["time_of_day"] = accidents["hour"].map(_classify_time_of_day)
    else:
        print("Warning: 'hour' column not found, skipping time_of_day")
    
    return accidents


def plot_accident_quality_overview(
    accidents_df: pd.DataFrame, 
    figsize: tuple = (14, 10),
    use_tueplots: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Generate comprehensive quality check and distribution overview for accident data.
    
    Creates a 3x3 grid of plots showing:
    - Temporal trends and patterns
    - Distribution across time dimensions (hour, weekday, month)
    - Missing values analysis
    - Severity distribution
    - Data quality validation checks
    
    Parameters
    ----------
    accidents_df : pd.DataFrame
        Accident dataframe with temporal columns (year, month, hour, weekday, etc.)
    figsize : tuple, optional
        Figure size (width, height), by default (16, 12)
    use_tueplots : bool, optional
        Whether to use tueplots ICML2024 stylesheet, by default True
    save_path : str | Path | None, optional
        Path to save the figure (e.g., "figname_icml.pdf"), by default None
    """
    # Apply tueplots styling if requested
    if use_tueplots:
        plt.rcParams.update(bundles.icml2024(column="full", nrows=3, ncols=3))
        # Use consistent color palette from tueplots
        colors = palettes.tue_plot
    else:
        # Fallback colors if tueplots not used
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Main color for consistency
    main_color = colors[0]
    accent_color = colors[1]
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle('', 
                 fontsize=20, fontweight='bold', y=0.985)

    # 1. Temporal trend: Accidents over time (year-month)
    ax1 = axes[0, 0]
    temporal_counts = accidents_df.groupby(['year', 'month']).size().reset_index(name='count')
    temporal_counts['date'] = pd.to_datetime(temporal_counts[['year', 'month']].assign(day=1))
    ax1.plot(temporal_counts['date'], temporal_counts['count'], marker='o', linewidth=2.5, color=main_color, markersize=5)
    ax1.set_title('Accidents Over Time (Monthly)', fontweight='bold', fontsize=13)
    ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Accidents', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45, labelsize=11)
    ax1.tick_params(axis='y', labelsize=11)

    # 2. Distribution by year
    ax2 = axes[0, 1]
    year_counts = accidents_df['year'].value_counts().sort_index()
    ax2.bar(year_counts.index, year_counts.values, color=main_color, edgecolor='black', linewidth=0.5)
    ax2.set_title('Accidents by Year', fontweight='bold', fontsize=13)
    ax2.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(labelsize=11)
    for i, v in enumerate(year_counts.values):
        ax2.text(year_counts.index[i], v + 50, str(v), ha='center', fontsize=10, fontweight='bold')

    # 3. Light conditions distribution
    ax3 = axes[0, 2]
    if 'light_condition' in accidents_df.columns:
        light_map = {0: 'Daylight', 1: 'Twilight', 2: 'Darkness'}
        light_counts = accidents_df['light_condition'].value_counts().sort_index()
        light_labels = [light_map.get(i, f'Code {i}') for i in light_counts.index]
        ax3.bar(light_labels, light_counts.values, color=main_color, edgecolor='black', linewidth=0.5)
        ax3.set_title('Accidents by Light Condition', fontweight='bold', fontsize=13)
        ax3.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax3.tick_params(labelsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(light_counts.values):
            ax3.text(i, v + 50, str(v), ha='center', fontsize=10, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'light_condition not available', ha='center', va='center', fontsize=11)
        ax3.set_title('Light Condition', fontweight='bold', fontsize=12)

    # 4. Road condition flag distribution
    ax4 = axes[1, 0]
    if 'road_condition' in accidents_df.columns:
        surface_map = {0: 'Dry', 1: 'Wet/Damp', 2: 'Slippery\n(Winter)'}
        surface_counts = accidents_df['road_condition'].value_counts().sort_index()
        surface_labels = [surface_map.get(i, f'Code {i}') for i in surface_counts.index]
        ax4.bar(surface_labels, surface_counts.values, color=main_color, edgecolor='black', linewidth=0.5)
        ax4.set_title('Accidents by Road Condition', fontweight='bold', fontsize=13)
        ax4.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax4.tick_params(labelsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(surface_counts.values):
            ax4.text(i, v + 50, str(v), ha='center', fontsize=10, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'road_condition not available', ha='center', va='center', fontsize=11)
        ax4.set_title('Road Condition', fontweight='bold', fontsize=12)

    # 5. Hourly distribution
    ax5 = axes[1, 1]
    if 'hour' in accidents_df.columns:
        hour_counts = accidents_df['hour'].value_counts().sort_index()
        ax5.bar(hour_counts.index, hour_counts.values, color=main_color, edgecolor='black', linewidth=0.5)
        ax5.set_title('Accidents by Hour of Day', fontweight='bold', fontsize=13)
        ax5.set_xlabel('Hour', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_xticks(range(0, 24, 2))
        ax5.tick_params(labelsize=11)
    else:
        ax5.text(0.5, 0.5, 'hour not available', ha='center', va='center', fontsize=11)
        ax5.set_title('Accidents by Hour of Day', fontweight='bold', fontsize=12)

    # 6. Day of week distribution (starting from Monday)
    ax6 = axes[1, 2]
    if 'weekday' in accidents_df.columns:
        # Remap: 2=Mon, 3=Tue, 4=Wed, 5=Thu, 6=Fri, 7=Sat, 1=Sun
        weekday_map = {2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat', 1: 'Sun'}
        weekday_order = [2, 3, 4, 5, 6, 7, 1]  # Monday to Sunday
        weekday_counts = accidents_df['weekday'].value_counts()
        # Reorder to start from Monday
        ordered_labels = [weekday_map[i] for i in weekday_order if i in weekday_counts.index]
        ordered_values = [weekday_counts[i] for i in weekday_order if i in weekday_counts.index]
        # Highlight weekends (Saturday=7, Sunday=1)
        colors_wd = [accent_color if i in [1, 7] else main_color for i in weekday_order if i in weekday_counts.index]
        ax6.bar(ordered_labels, ordered_values, color=colors_wd, edgecolor='black', linewidth=0.5)
        ax6.set_title('Accidents by Day of Week', fontweight='bold', fontsize=13)
        ax6.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.tick_params(labelsize=11)
    else:
        ax6.text(0.5, 0.5, 'weekday not available', ha='center', va='center', fontsize=11)
        ax6.set_title('Accidents by Day of Week', fontweight='bold', fontsize=12)

    # 7. Spatial Distribution Heatmap (sample of Berlin)
    ax7 = axes[2, 0]
    
    # Try to get coordinates from geometry or raw coordinate columns
    x_coords = None
    y_coords = None
    
    if 'geometry' in accidents_df.columns:
        try:
            x_coords = accidents_df.geometry.x
            y_coords = accidents_df.geometry.y
        except:
            pass
    
    # Fallback to raw coordinate columns (WGS84)
    if x_coords is None and 'XGCSWGS84' in accidents_df.columns and 'YGCSWGS84' in accidents_df.columns:
        x_coords = accidents_df['XGCSWGS84']
        y_coords = accidents_df['YGCSWGS84']
    
    if x_coords is not None and y_coords is not None:
        try:
            # Remove NaN values
            valid_mask = x_coords.notna() & y_coords.notna()
            x_valid = x_coords[valid_mask]
            y_valid = y_coords[valid_mask]
            
            if len(x_valid) > 0:
                # Create 2D histogram (heatmap)
                hist, xedges, yedges = np.histogram2d(
                    x_valid, y_valid, bins=50
                )
                
                # Plot heatmap with better colormap (YlOrRd: yellow-orange-red without black)
                im = ax7.imshow(
                    hist.T, origin='lower', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='YlOrRd', aspect='auto', interpolation='bilinear',
                    vmin=0
                )
                
                # Add contour lines for better definition
                from matplotlib import pyplot as plt_contour
                X, Y = np.meshgrid(
                    (xedges[:-1] + xedges[1:]) / 2,
                    (yedges[:-1] + yedges[1:]) / 2
                )
                # Only show contours for non-zero areas
                levels = np.linspace(hist[hist > 0].min() if (hist > 0).any() else 1, 
                                    hist.max(), 5)
                ax7.contour(X, Y, hist.T, levels=levels, colors='darkred', 
                           linewidths=0.5, alpha=0.4)
                
                ax7.set_title('Spatial Distribution (Berlin)', fontweight='bold', fontsize=12)
                ax7.set_xlabel('Longitude' if 'XGCSWGS84' in accidents_df.columns else 'Easting (m)', 
                              fontsize=11, fontweight='bold')
                ax7.set_ylabel('Latitude' if 'YGCSWGS84' in accidents_df.columns else 'Northing (m)', 
                              fontsize=11, fontweight='bold')
                ax7.tick_params(labelsize=9)
                
                # Format tick labels
                ax7.ticklabel_format(style='plain', axis='both')
                
                # Add colorbar with better styling
                from matplotlib.pyplot import colorbar
                cbar = colorbar(im, ax=ax7)
                cbar.set_label('Accident Count', fontsize=10)
                cbar.ax.tick_params(labelsize=9)
            else:
                ax7.text(0.5, 0.5, 'No valid coordinates', 
                        ha='center', va='center', fontsize=11)
                ax7.set_title('Spatial Distribution', fontweight='bold', fontsize=12)
        except Exception as e:
            ax7.text(0.5, 0.5, f'Heatmap unavailable:\n{str(e)[:50]}', 
                    ha='center', va='center', fontsize=10)
            ax7.set_title('Spatial Distribution', fontweight='bold', fontsize=12)
    else:
        ax7.text(0.5, 0.5, 'Coordinate data not available', 
                ha='center', va='center', fontsize=11)
        ax7.set_title('Spatial Distribution', fontweight='bold', fontsize=12)

    # 8. Injury severity distribution
    ax8 = axes[2, 1]
    if 'injury_severity' in accidents_df.columns:
        severity_counts = accidents_df['injury_severity'].value_counts().sort_index()
        severity_map = {1: 'Killed', 2: 'Seriously\ninjured', 3: 'Slightly\ninjured'}
        severity_labels = [severity_map.get(i, f'Code {i}') for i in severity_counts.index]
        ax8.bar(severity_labels, severity_counts.values, 
                color=main_color, edgecolor='black', linewidth=0.5)
        ax8.set_title('Accident Severity Distribution', fontweight='bold', fontsize=13)
        ax8.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.tick_params(labelsize=11)
        for i, v in enumerate(severity_counts.values):
            ax8.text(i, v + 50, str(v), ha='center', fontsize=10, fontweight='bold')
        # Add legend
        legend_text = '1=Killed, 2=Seriously injured, 3=Slightly injured'
        ax8.text(0.5, -0.25, legend_text, transform=ax8.transAxes, ha='center', 
                fontsize=9, style='italic')
    else:
        ax8.text(0.5, 0.5, 'injury_severity not available', ha='center', va='center', fontsize=11)
        ax8.set_title('Accident Severity Distribution', fontweight='bold', fontsize=12)

    # 9. Month distribution (seasonality check) - All same color
    ax9 = axes[2, 2]
    if 'month' in accidents_df.columns:
        month_counts = accidents_df['month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax9.bar([month_names[i-1] for i in month_counts.index], month_counts.values, 
                color=main_color, edgecolor='black', linewidth=0.5)
        ax9.set_title('Accidents by Month (Seasonality)', fontweight='bold', fontsize=13)
        ax9.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax9.tick_params(axis='x', rotation=45, labelsize=11)
        ax9.tick_params(axis='y', labelsize=11)
        ax9.grid(True, alpha=0.3, axis='y')
    else:
        ax9.text(0.5, 0.5, 'month not available', ha='center', va='center', fontsize=11)
        ax9.set_title('Accidents by Month (Seasonality)', fontweight='bold', fontsize=12)

    # Adjust layout with more spacing between rows
    fig.subplots_adjust(top=0.94, hspace=0.4, wspace=0.3)
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
    
    plt.show()

    # Print summary statistics
    print("\n" + "="*28 + "SUMMARY STATISTICS" + "="*28)
    print(f"Total accidents: {len(accidents_df):,}")
    print(f"Date range: {accidents_df['year'].min()} - {accidents_df['year'].max()}")
    print(f"Number of columns: {len(accidents_df.columns)}")
    
    # Duplicate check using accident_id_extended (excluding NaN)
    if 'accident_id_extended' in accidents_df.columns:
        # Count only non-NaN values
        non_null_ids = accidents_df['accident_id_extended'].dropna()
        n_total_non_null = len(non_null_ids)
        n_unique = non_null_ids.nunique()
        n_duplicates = n_total_non_null - n_unique
        n_null = accidents_df['accident_id_extended'].isna().sum()
        status = '[PASS]' if n_duplicates == 0 else '[FAIL]'
        print(f"\n{status} Duplicate check (accident_id_extended)")
        print(f"  - Total non-null records: {n_total_non_null:,}")
        print(f"  - Unique IDs: {n_unique:,}")
        print(f"  - Duplicates: {n_duplicates}")
        print(f"  - Missing IDs (NaN): {n_null:,}")
    
    # Temporal validity check
    if 'year' in accidents_df.columns:
        min_year = 2019  # Expected earliest year
        max_year = 2023  # Expected latest year
        past_dates = (accidents_df['year'] < min_year).sum()
        future_dates = (accidents_df['year'] > max_year).sum()
        temporal_status = '[PASS]' if (past_dates == 0 and future_dates == 0) else '[FAIL]'
        print(f"\n{temporal_status} Temporal validity check ({min_year}-{max_year})")
        if past_dates > 0:
            print(f"  - Records before {min_year}: {past_dates}")
        if future_dates > 0:
            print(f"  - Records after {max_year}: {future_dates}")
        if temporal_status == '[PASS]':
            print(f"  - All records within expected range ({min_year}-{max_year})")
    
    # Coordinate bounds check (Berlin)
    if 'geometry' in accidents_df.columns:
        min_x, max_x = 370000, 415000
        min_y, max_y = 5800000, 5840000
        try:
            out_of_bounds = (
                (accidents_df.geometry.x < min_x) | (accidents_df.geometry.x > max_x) |
                (accidents_df.geometry.y < min_y) | (accidents_df.geometry.y > max_y)
            ).sum()
            coord_status = '[PASS]' if out_of_bounds == 0 else '[FAIL]'
            print(f"\n{coord_status} Coordinate bounds check (Berlin)")
            if out_of_bounds > 0:
                print(f"  - Records outside Berlin bounds: {out_of_bounds}")
            else:
                print(f"  - All coordinates within Berlin (EPSG:32633)")
        except Exception as e:
            print(f"\nCoordinate bounds check: ERROR - {str(e)}")
    
    # Missing values in accident features and coordinates only
    print(f"\nMissing values in accident features and geographic columns:")
    
    # Define accident and geo columns
    accident_feature_cols = [
        'year', 'month', 'hour', 'weekday', 'weekday_type', 'time_of_day',
        'injury_severity', 'accident_kind', 'accident_type', 'light_condition',
        'car_involved', 'pedestrian_involved',
        'motorcycle_involved', 'goods_vehicle_involved', 'other_vehicle_involved',
        'road_condition'
    ]
    
    geo_cols = ['LINREFX', 'LINREFY', 'XGCSWGS84', 'YGCSWGS84']
    
    all_feature_cols = accident_feature_cols + geo_cols
    all_feature_cols = [c for c in all_feature_cols if c in accidents_df.columns]
    
    missing_by_col = accidents_df[all_feature_cols].isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    
    if len(missing_by_col) > 0:
        for col, count in missing_by_col.items():
            pct = count / len(accidents_df) * 100
            print(f"  {col}: {count:,} ({pct:.1f}%)")
    else:
        print(f"  No missing values in accident features and geographic columns!")

