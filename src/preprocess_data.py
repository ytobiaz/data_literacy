from pathlib import Path
import pandas as pd

def preprocess_accident_data(save_to_parquet: bool = False):
    csv_dir = Path(Path(__file__).parent.parent / "data" / "csv")
    csv_files = sorted(csv_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir.resolve()}")

    dfs = []
    for fp in csv_files:
        df = pd.read_csv(fp, low_memory=False, delimiter=";", decimal=",")
        # since column names are inconsistent, rename "IstStrasse" or "STRZUSTAND" to "IstStrassenzustand" and "IstSonstig" to "IstSonstige" if needed
        if "IstStrasse" in df.columns and "IstStrassenzustand" not in df.columns:
            df.rename(columns={"IstStrasse": "IstStrassenzustand"}, inplace=True)
        if "STRZUSTAND" in df.columns and "IstStrassenzustand" not in df.columns:
            df.rename(columns={"STRZUSTAND": "IstStrassenzustand"}, inplace=True)
        if "IstSonstig" in df.columns and "IstSonstige" not in df.columns:
            df.rename(columns={"IstSonstig": "IstSonstige"}, inplace=True)
        df["source_file"] = fp.name 
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # parse values of "ULICHTVERH" and "IstGkfz" as Integers (Why are they not already read as such?)
    df_all["ULICHTVERH"] = pd.to_numeric(df_all["ULICHTVERH"], errors="coerce").astype("Int64")
    df_all["IstGkfz"] = pd.to_numeric(df_all["IstGkfz"], errors="coerce").astype("Int64")
    print(f"Loaded {len(csv_files)} files -> combined shape: {df_all.shape}")

    # drop columns that are not relevant or don't contain information
    df_all.drop(columns=["OBJECTID", "UIDENTSTLA", "OBJECTID_1", "UIDENTSTLAE", "OID_", "FID", "LICHT", "PLST"], inplace=True)
    print(f"Dropped irrelevant columns -> shape: {df_all.shape}")

    # drop all accidents that did not involve bicycles (column 'IstRad' != 1)
    df_bike = df_all[df_all['IstRad'] == 1].copy()
    print(f"Filtered to bicycle accidents -> shape: {df_bike.shape}")

    # only keep accidents in Berlin (column 'ULAND' == 11) and drop more irrelevant columns
    df_bike_berlin = df_bike[df_bike['ULAND'] == 11].copy()
    df_bike_berlin = df_bike_berlin.drop(columns=['ULAND', 'UREGBEZ', 'UKREIS', 'UGEMEINDE', 'IstRad'])
    df_bike_berlin = df_bike_berlin.reset_index(drop=True)

    # assign english column names
    df_bike_berlin.rename(columns={
        "UJAHR": "year",
        "UMONAT": "month",
        "USTUNDE": "hour",
        "UWOCHENTAG": "weekday",
        "UKATEGORIE": "injury_severity",
        "UART": "accident_kind",
        "UTYP1": "accident_type",
        "IstPKW": "car_involved",
        "IstFuss": "pedestrian_involved",
        "IstKrad": "motorcycle_involved",
        "IstSonstige": "other_vehicle_involved",
        "IstGkfz": "goods_vehicle_involved",
        "IstStrassenzustand": "road_condition",
        "ULICHTVERH": "light_condition"
    }, inplace=True)
    
    print(f"Filtered to bicycle accidents in Berlin -> shape: {df_bike_berlin.shape}")
    df_bike_berlin.head()
    if save_to_parquet:
        # save preprocessed data to parquet
        output_fp = Path(Path(__file__).parent.parent / "data" / "preprocessed" / "bicycle_accidents_berlin.parquet")
        df_bike_berlin.to_parquet(output_fp, index=False)   
    return df_bike_berlin
