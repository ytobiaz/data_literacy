from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import os

import geopandas as gpd
from geopandas import GeoDataFrame
import shapely.wkt
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar

from .utils import data_path


@dataclass(frozen=True)
class SegmentGeometry:
    segments_gdf: GeoDataFrame
    segment_static: GeoDataFrame
    canonical_crs: str


def load_segment_geometry(
    parquet_path: str | Path | None = None,
    *,
    canonical_crs: str = "EPSG:32633",
    source_crs: str = "EPSG:4326",
) -> SegmentGeometry:
    """Load canonical Strava segment geometries and reproject to a metric CRS."""

    parquet_path = (
        Path(parquet_path) if parquet_path is not None else data_path("strava", "berlin_graph_geometry.parquet")
    )
    
    if not parquet_path.exists():
        # If we don't have a local copy, download from Zenodo (original source).
        base_url = "https://zenodo.org/records/15332147/files"
        url = f"{base_url}/berlin_graph_geometry.parquet?download=1"
        print(f"Downloading berlin_graph_geometry.parquet from Zenodo...")
        df = pd.read_parquet(url, engine="pyarrow")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path)
        print(f"Saved berlin_graph_geometry.parquet to {parquet_path}")
    else:
        print(f"Reading berlin_graph_geometry.parquet locally from {parquet_path}")
        df = pd.read_parquet(parquet_path)

    segment_geo_df = pd.read_parquet(parquet_path).copy()
    if "geometry" not in segment_geo_df.columns:
        raise KeyError("Expected a 'geometry' column in the segment geometry parquet")

    # Convert WKT geometry strings to shapely objects
    segment_geo_df["geometry"] = segment_geo_df["geometry"].apply(shapely.wkt.loads)

    # Create GeoDataFrame and reproject to canonical CRS
    segment_geo_gdf = gpd.GeoDataFrame(
        segment_geo_df,
        geometry="geometry",
        crs=source_crs,
    ).to_crs(canonical_crs)

    # Prepare static segment attributes
    cols_static = ["counter_name", "geometry"]
    if "street_name" in segment_geo_gdf.columns:
        cols_static.append("street_name")

    segment_static = (
        segment_geo_gdf[cols_static]
        .drop_duplicates("counter_name")
        .reset_index(drop=True)
    )

    return SegmentGeometry(
        segments_gdf=segment_geo_gdf,
        segment_static=segment_static,
        canonical_crs=canonical_crs,
    )


def plot_segment_quality_overview(
    segments_gdf,
    figsize=(6, 6),
    use_tueplots=True,
    save_path=None,
):
    """
    Create comprehensive quality check and overview visualization for road network segment data.
    
    Parameters
    ----------
    segments_gdf : GeoDataFrame
        Road network segments with geometry, counter_name, latitude, longitude
    figsize : tuple, optional
        Figure size (width, height), by default (6, 6)
    use_tueplots : bool, optional
        Whether to use tueplots ICML2024 stylesheet, by default True
    save_path : str | Path | None, optional
        Path to save the figure, by default None
    """
    # Apply tueplots styling if requested
    if use_tueplots:
        from tueplots import bundles
        from tueplots.constants.color import palettes
        plt.rcParams.update(bundles.icml2024(column="full", nrows=1, ncols=2))
        colors = palettes.tue_plot
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    main_color = colors[0]
    accent_color = colors[1]
    
    width = 8
    figheight = width / 2
    fig, axes = plt.subplots(1, 2, figsize=(width, figheight))
    # Ensure axes is a flat array for 1x2 layout
    ax1, ax2 = axes.flatten() if isinstance(axes, np.ndarray) else (axes[0], axes[1])

    for _ax in (ax1, ax2):
        if _ax is not None:
            _ax.set_box_aspect(1)
    
    # 1. Segment length distribution
    if 'geometry' in segments_gdf.columns:
        segment_lengths = segments_gdf.geometry.length
        ax1.hist(segment_lengths, bins=50, color=main_color, edgecolor='black', linewidth=0.5)
        ax1.set_title('Segment Length Distribution', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Length (m)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(labelsize=10)
        # Add statistics
        median_len = segment_lengths.median()
        ax1.axvline(median_len, color=accent_color, linestyle='--', linewidth=2, 
                   label=f'Median: {median_len:.1f}m')
        ax1.legend(fontsize=10)
    else:
        ax1.text(0.5, 0.5, 'geometry not available', ha='center', va='center', fontsize=11)
        ax1.set_title('Segment Length Distribution', fontweight='bold', fontsize=13)
    
    # 2. Detailed street network map
    if 'geometry' in segments_gdf.columns and len(segments_gdf) > 0:
        try:
            # Convert to WGS84 for proper lat/lon display if in projected CRS
            if segments_gdf.crs and segments_gdf.crs.is_projected:
                segments_display = segments_gdf.to_crs(epsg=4326)
            else:
                segments_display = segments_gdf
            
            # Plot all segments with thin lines to show street network detail
            segments_display.plot(ax=ax2, color=main_color, linewidth=0.8, alpha=0.7)
            ax2.set_title('Street Network Detail', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Longitude', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Latitude', fontsize=11, fontweight='bold')
            ax2.tick_params(labelsize=10)
            # Format coordinates properly (e.g., 52.5 instead of 52.500000)
            from matplotlib.ticker import FormatStrFormatter
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # Use adjustable aspect ratio for geographic coordinates (lat/lon have different scales)
            ax2.set_aspect('auto')
            ax2.grid(True, alpha=0.2, linestyle='--')
            # Add segment count annotation
            ax2.text(0.05, 0.95, f'n={len(segments_gdf):,}', transform=ax2.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            ax2.text(0.5, 0.5, f'Map error:\n{str(e)[:30]}', ha='center', va='center', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'geometry not available', ha='center', va='center', fontsize=11)
        ax2.set_title('Street Network Detail', fontweight='bold', fontsize=13)
    
    # Removed long segments map per request
    
    # Store validation results for summary (not plotted)
    validation_results = {}
    
    # Validation Test 1: All geometries valid
    if 'geometry' in segments_gdf.columns:
        invalid_geom = (~segments_gdf.geometry.is_valid).sum()
        validation_results['Valid Geometry'] = (invalid_geom == 0, f'All geometries are valid (invalid: {invalid_geom})')
    
    # Validation Test 2: No empty geometries
    if 'geometry' in segments_gdf.columns:
        empty_geom = segments_gdf.geometry.is_empty.sum()
        validation_results['No Empty Geometry'] = (empty_geom == 0, f'No empty geometries (empty: {empty_geom})')
    
    # Validation Test 3: Coordinate range check (Berlin area)
    if 'latitude' in segments_gdf.columns and 'longitude' in segments_gdf.columns:
        lat_valid = segments_gdf['latitude'].notna()
        lon_valid = segments_gdf['longitude'].notna()
        lat_range_ok = ((segments_gdf.loc[lat_valid, 'latitude'] >= 52.3) & 
                       (segments_gdf.loc[lat_valid, 'latitude'] <= 52.7)).all()
        lon_range_ok = ((segments_gdf.loc[lon_valid, 'longitude'] >= 13.0) & 
                       (segments_gdf.loc[lon_valid, 'longitude'] <= 13.8)).all()
        validation_results['Coordinate Range (Berlin)'] = (lat_range_ok and lon_range_ok, 
                                                           f'Coordinates within Berlin bounds (lat: 52.3-52.7, lon: 13.0-13.8)')
    
    # Validation Test 4: No duplicate segment names
    if 'counter_name' in segments_gdf.columns:
        n_segments = len(segments_gdf)
        n_unique = segments_gdf['counter_name'].nunique()
        validation_results['Unique Segment IDs'] = (n_segments == n_unique, 
                                                    f'All segment IDs are unique (total: {n_segments}, unique: {n_unique})')
    
    # Validation Test 5: Reasonable segment lengths (not too short/long)
    if 'geometry' in segments_gdf.columns:
        lengths = segments_gdf.geometry.length
        too_short = (lengths < 1).sum()
        too_long = (lengths > 5000).sum()
        reasonable_lengths = (too_short == 0 and too_long == 0)
        validation_results['Reasonable Lengths (1-5000m)'] = (reasonable_lengths, 
                                                              f'All lengths in valid range (too short: {too_short}, too long: {too_long})')
    
    # Adjust layout (skip when tueplots enables constrained layout)
    if not use_tueplots:
        fig.subplots_adjust(hspace=0.3, wspace=0.35)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*28 + "SUMMARY STATISTICS" + "="*28)
    print(f"Total segments: {len(segments_gdf):,}")
    print(f"Number of columns: {len(segments_gdf.columns)}")
    
    # Data Quality Checks
    for test_name, (result, description) in validation_results.items():
        status = 'PASS' if result else 'FAIL'
        print(f"\n[{status}] {test_name}")
        print(f"  Description: {description}")
    
    # Missing values
    missing_by_col = segments_gdf.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    
    if len(missing_by_col) > 0:
        print(f"\nMissing values by column:")
        for col, count in missing_by_col.items():
            pct = count / len(segments_gdf) * 100
            print(f"  {col}: {count:,} ({pct:.1f}%)")
    else:
        print(f"\nNo missing values detected!")
    
