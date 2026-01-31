import pandas as pd
import geopandas as gpd
from shapely import wkt


def load_counting_stations(excel_path):
    """Load and combine official bike counting station data from Excel (2019-2023). Returns tuple of (station_metadata, hourly_counts)."""
    xls = pd.ExcelFile(excel_path)

    # Load station metadata (location, installation date, etc.)
    counting_stations_location = pd.read_excel(xls, sheet_name='Standortdaten')

    # Rename columns to standardized English names
    counting_stations_location = counting_stations_location.rename(columns={
        "Zählstelle": "station_id",
        "Beschreibung - Fahrtrichtung": "description",
        "Breitengrad": "lat",
        "Längengrad": "lon",
        "Installationsdatum": "installed"
    })

    # Select the sheets corresponding to years 2019–2023
    # Sheet names follow the pattern: "Jahresdatei <year>"
    target_sheets = [
        s for s in xls.sheet_names
        if s.startswith("Jahresdatei")
        and 2019 <= int(s.split()[-1]) <= 2023
    ]

    counting_stations = []

    # Load and clean each yearly sheet
    for sheet in target_sheets:
        year = int(sheet.split()[-1])
        print(f"Reading {sheet}...")

        # Read entire sheet as strings 
        df = pd.read_excel(
            xls,
            sheet_name=sheet,
            skiprows=0,   # No extra header rows in this dataset
            dtype=str
        )

        # Clean and convert column types

        # Convert hourly count ("Wert") to float (German → international format)
        if "Wert" in df.columns:
            df["Wert"] = (
                df["Wert"]
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
        
        # Convert hour column ("Stunde") to integer
        if "Stunde" in df.columns:
            df["Stunde"] = df["Stunde"].astype(int)
        
        # Convert date column ("Datum") to datetime
        if "Datum" in df.columns:
            df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
        
        # Create full datetime (date + hour)
        if "Datum" in df.columns and "Stunde" in df.columns:
            df["datetime"] = df["Datum"] + pd.to_timedelta(df["Stunde"], unit="h")

        # Add year extracted from sheet name
        df["year"] = year

        counting_stations.append(df)

    # Combine all yearly sheets into a single unified DataFrame
    counting_stations = pd.concat(counting_stations, ignore_index=True)
    counting_stations = counting_stations.rename(columns={
        "Zählstelle        Inbetriebnahme": "datetime"
    })
    
    return counting_stations_location, counting_stations


def build_station_strava_comparison_panel(counting_stations, matched, strava_berlin_data_raw):
    """Merge official counting station daily counts with Strava trip counts. Returns DataFrame with total_count (official) and strata_count (Strava) columns."""
    # Daily SUM per segment from station counts
    cs = counting_stations.copy()
    cs["datetime"] = pd.to_datetime(cs["datetime"], errors="coerce")

    long_daily = (
        cs.melt("datetime", var_name="station_col", value_name="count")
          .assign(
              station=lambda d: d["station_col"].str.split().str[0],
              date=lambda d: d["datetime"].dt.normalize(),
              count=lambda d: pd.to_numeric(d["count"], errors="coerce"),
          )
          .merge(
              matched[["station_id", "counter_name"]],
              left_on="station",
              right_on="station_id",
              how="left",
          )
          .dropna(subset=["counter_name"])
          .groupby(["counter_name", "date"], as_index=False)["count"]
          .sum()
          .rename(columns={"counter_name": "segment", "count": "total_count"})
    )

    # Add Strava daily counts (from existing raw table)
    strava_daily = strava_berlin_data_raw.copy()
    strava_daily["date"] = pd.to_datetime(strava_daily["date"]).dt.normalize()

    long_daily = (
        long_daily.merge(
            strava_daily[["date", "counter_name", "strava_total_trip_count"]],
            left_on=["segment", "date"],
            right_on=["counter_name", "date"],
            how="left",
        )
        .drop(columns="counter_name")
        .rename(columns={"strava_total_trip_count": "strata_count"})
    )
    
    return long_daily


def match_counting_stations_to_segments(segment_geo_gdf, counting_stations_location, target_crs=3857):
    """Match official counting stations to street segments via spatial nearest-neighbor join. Returns tuple of (matched, gdf_strava)."""
    # Strava geometry
    gdf_strava = segment_geo_gdf.copy()

    if len(gdf_strava) and isinstance(gdf_strava["geometry"].iloc[0], str):
        gdf_strava["geometry"] = gdf_strava["geometry"].apply(wkt.loads)

    gdf_strava = gdf_strava.set_geometry("geometry")

    if gdf_strava.crs is None:
        gdf_strava = gdf_strava.set_crs("EPSG:4326")

    gdf_strava = gdf_strava.to_crs(target_crs)

    # Counting stations
    gdf_stations = counting_stations_location.copy()

    gdf_stations = gpd.GeoDataFrame(
        gdf_stations,
        geometry=gpd.points_from_xy(gdf_stations["lon"], gdf_stations["lat"]),
        crs="EPSG:4326",
    ).to_crs(target_crs)

    # Nearest-neighbor match
    matched = gdf_stations.sjoin_nearest(
        gdf_strava[["geometry", "counter_name"]],
        how="left",
        distance_col="distance_m"
    )

    return matched, gdf_strava
