from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import geopandas as gpd
from geopandas import GeoDataFrame

from .utils import data_path


ACCIDENT_COLUMNS_EN: dict[str, str] = {
    # IDs & metadata
    "OBJECTID": "object_id",
    "OBJECTID_1": "object_id_alt",
    "OID_": "oid",
    "FID": "fid",
    "source_file": "source_file",

    # Unique accident identifiers
    "UIDENTSTLA": "accident_id",
    "UIDENTSTLAE": "accident_id_extended",

    # Administrative divisions
    "ULAND": "land_code",
    "UREGBEZ": "admin_region_code",
    "UKREIS": "district_code",
    "UGEMEINDE": "municipality_code",

    # Time
    "UJAHR": "year",
    "UMONAT": "month",
    "USTUNDE": "hour",
    "UWOCHENTAG": "weekday",

    # Accident classification
    "UKATEGORIE": "injury_severity",
    "UART": "accident_kind",
    "UTYP1": "accident_type",

    # Participants involved (0/1)
    "IstRad": "involved_bicycle",
    "IstPKW": "involved_passenger_car",
    "IstFuss": "involved_pedestrian",
    "IstKrad": "involved_motorcycle",
    "IstSonstig": "involved_other_vehicle_old",
    "IstGkfz": "involved_goods_vehicle",
    "IstSonstige": "involved_other_vehicle",
    "IstStrasse": "involved_road",
    "IstStrassenzustand": "road_condition_flag",

    # Environmental conditions
    "LICHT": "light_condition_old",
    "ULICHTVERH": "light_condition",
    "STRZUSTAND": "road_condition",

    # Data quality
    "PLST": "plausibility_level",
}


def _coerce_numeric_with_decimal_comma(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            df[col] = pd.to_numeric(series, errors="coerce")
            continue
        df[col] = (
            series.astype(str)
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
    return df


def load_accidents_raw(
    csv_dir: str | Path | None = None,
    *,
    delimiter: str = ";",
    decimal: str = ",",
    low_memory: bool = False,
) -> pd.DataFrame:
    """Load all Unfallatlas CSVs from `data/csv` (or a provided directory).

    Keeps original column names (German), and adds a `source_file` column.
    """

    csv_dir_path = Path(csv_dir) if csv_dir is not None else data_path("csv")
    csv_files = sorted(csv_dir_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir_path.resolve()}")

    dfs: list[pd.DataFrame] = []
    for fp in csv_files:
        df = pd.read_csv(fp, low_memory=low_memory, delimiter=delimiter, decimal=decimal)

        # Some years use slightly different column names.
        if "IstStrasse" in df.columns and "IstStrassenzustand" not in df.columns:
            df = df.rename(columns={"IstStrasse": "IstStrassenzustand"})
        if "STRZUSTAND" in df.columns and "IstStrassenzustand" not in df.columns:
            df = df.rename(columns={"STRZUSTAND": "IstStrassenzustand"})
        if "IstSonstig" in df.columns and "IstSonstige" not in df.columns:
            df = df.rename(columns={"IstSonstig": "IstSonstige"})

        df["source_file"] = fp.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def prepare_accidents_bike_berlin(
    accidents_raw: pd.DataFrame,
    *,
    column_map: Mapping[str, str] = ACCIDENT_COLUMNS_EN,
    berlin_land_code: int = 11,
) -> pd.DataFrame:
    """Rename columns to English and filter to bicycle accidents in Berlin."""

    accidents = accidents_raw.rename(columns=dict(column_map))

    required_cols = {"involved_bicycle", "land_code"}
    missing = required_cols - set(accidents.columns)
    if missing:
        raise KeyError(f"Missing required columns after renaming: {sorted(missing)}")

    accidents_bike_berlin = (
        accidents.loc[
            (accidents["involved_bicycle"] == 1) & (accidents["land_code"] == berlin_land_code)
        ]
        .reset_index(drop=True)
    )

    # Make common coordinate columns numeric (robust to comma decimals / strings).
    _coerce_numeric_with_decimal_comma(
        accidents_bike_berlin,
        cols=["XGCSWGS84", "YGCSWGS84", "LINREFX", "LINREFY"],
    )

    return accidents_bike_berlin


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
