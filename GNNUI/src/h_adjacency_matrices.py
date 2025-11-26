import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging
import paths as paths
from math import radians, cos, sin, asin, sqrt
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import networkx as nx
from paths import *


def load_file(file_path):
    with open(file_path, "rb") as f:
        return pkl.load(f)


def save_file(file_path, data):
    with open(file_path, "wb") as f:
        pkl.dump(data, f)


def calculate_random_walk_matrix(adj_mx):
    # logging.info("Calculating random walk matrix...")
    if isinstance(adj_mx, torch.Tensor):
        adj_mx = adj_mx.cpu().numpy()  # Convert to NumPy array if it's a tensor
    adj_mx = sp.coo_matrix(adj_mx)  # Convert to sparse matrix
    d = np.array(adj_mx.sum(1)).flatten()  # Sum of rows and flatten to 1D array
    d = d.astype(float)  # Convert to float to handle inversion
    d_inv = np.power(d, -1, where=d != 0)  # Invert the non-zero elements only
    d_inv[np.isinf(d_inv)] = (
        0.0  # Replace infinities with zeros (in case of division by zero)
    )
    d_mat_inv = sp.diags(d_inv)  # Create a diagonal matrix of inverses
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()  # Compute random walk matrix
    # logging.info("Random walk matrix calculation completed.")
    return random_walk_mx.toarray()


def get_file_paths(matrix_type, target_feature):
    """
    Get file paths for adjacency matrices based on matrix type and target feature.

    Parameters:
    - matrix_type (str): Type of adjacency matrix.
    - target_feature (str): Target feature (e.g., "strava" or "taxi").

    Returns:
    - tuple: Paths for the primary and secondary adjacency matrices.
    """
    base_paths = {
        "strava": {
            "binary": (BERLIN_ADJACENCY_BINARY, BERLIN_ADJACENCY_BINARY),
            "distance_birdfly": (BERLIN_ADJACENCY_BIRD, BERLIN_ADJACENCY_BIRD),
            "distance_traveltime": (BERLIN_ADJACENCY_TRAVELTIME, BERLIN_ADJACENCY_TRAVELTIME),
            "distance_roaddistance": (BERLIN_ADJACENCY_ROADDISTANCE, BERLIN_ADJACENCY_ROADDISTANCE),
            "similarity": (BERLIN_ADJACENCY_SIMILARITY, BERLIN_ADJACENCY_SIMILARITY),
            "binary_and_distance_birdfly": (BERLIN_ADJACENCY_BINARY, BERLIN_ADJACENCY_BIRD),
            "binary_and_distance_traveltime": (BERLIN_ADJACENCY_BINARY, BERLIN_ADJACENCY_TRAVELTIME),
            "binary_and_distance_roaddistance": (BERLIN_ADJACENCY_BINARY, BERLIN_ADJACENCY_ROADDISTANCE),
            "binary_and_similarity": (BERLIN_ADJACENCY_BINARY, BERLIN_ADJACENCY_SIMILARITY),
            "similarity_and_distance_traveltime": (BERLIN_ADJACENCY_SIMILARITY, BERLIN_ADJACENCY_TRAVELTIME),
            "similarity_and_distance_roaddistance": (BERLIN_ADJACENCY_SIMILARITY, BERLIN_ADJACENCY_ROADDISTANCE),
            "similarity_and_distance_birdfly": (BERLIN_ADJACENCY_SIMILARITY, BERLIN_ADJACENCY_BIRD),
        },
        "taxi": {
            "binary": (NY_ADJACENCY_BINARY, NY_ADJACENCY_BINARY),
            "distance_birdfly": (NY_ADJACENCY_BIRD, NY_ADJACENCY_BIRD),
            "distance_traveltime": (NY_ADJACENCY_TRAVELTIME, NY_ADJACENCY_TRAVELTIME),
            "distance_roaddistance": (NY_ADJACENCY_ROADDISTANCE, NY_ADJACENCY_ROADDISTANCE),
            "similarity": (NY_ADJACENCY_SIMILARITY, NY_ADJACENCY_SIMILARITY),
            "binary_and_distance_birdfly": (NY_ADJACENCY_BINARY, NY_ADJACENCY_BIRD),
            "binary_and_distance_traveltime": (NY_ADJACENCY_BINARY, NY_ADJACENCY_TRAVELTIME),
            "binary_and_distance_roaddistance": (NY_ADJACENCY_BINARY, NY_ADJACENCY_ROADDISTANCE),
            "binary_and_similarity": (NY_ADJACENCY_BINARY, NY_ADJACENCY_SIMILARITY),
            "similarity_and_distance_traveltime": (NY_ADJACENCY_SIMILARITY, NY_ADJACENCY_TRAVELTIME),
            "similarity_and_distance_roaddistance": (NY_ADJACENCY_SIMILARITY, NY_ADJACENCY_ROADDISTANCE),
            "similarity_and_distance_birdfly": (NY_ADJACENCY_SIMILARITY, NY_ADJACENCY_BIRD),
        },
    }

    target_key = (
        "strava"
        if "strava" in target_feature
        else "taxi" if "taxi" in target_feature else None
    )
    if target_key is None or matrix_type not in base_paths[target_key]:
        raise ValueError(
            f"Invalid combination of matrix_type='{matrix_type}' and target_feature='{target_feature}'."
        )

    return base_paths[target_key][matrix_type]


def load_adjacency_matrix_and_counter_ids(
    feature_selection, type_of_adjacency_matrix, target_feature
):

    if (
        feature_selection == "limited_all_features_for_strava_as_target"
        or feature_selection == "full_data_all_features_for_strava_as_target"
    ):
        file_to_load_pkl = GNNUI_DATA_STRAVA
    elif (
        feature_selection == "limited_all_features_for_taxi_as_target"
        or feature_selection == "full_data_all_features_for_taxi_as_target"
    ):
        file_to_load_pkl = GNNUI_DATA_TAXI
    else:
        raise ValueError("This type of feature selection is not implemented")

    ############## Generate A and counterIDs
    # if A and counter_ids are already generated, we do not need to generate them again, then we just load them
    A_file_path_primary, A_file_path_secondary = get_file_paths(type_of_adjacency_matrix, target_feature)

    if os.path.exists(A_file_path_primary) and os.path.exists(
        A_file_path_secondary
    ):
        # if and '_and_' is in the file name, we load the two adjacency matrices
        if "_and_" in type_of_adjacency_matrix: 
            A_primary = pd.read_parquet(A_file_path_primary)
            A_secondary = A_primary.copy()
            assert (
                A_primary.columns.tolist() == A_secondary.columns.tolist()
            )
        else:   
            A_primary = pd.read_parquet(A_file_path_primary)
            A_secondary = pd.read_parquet(A_file_path_secondary)
        counter_ids = A_primary.columns.tolist()
        # Turn all elements in A to float
        if type_of_adjacency_matrix == "binary":
            A_primary = A_primary.astype(float)
            A_secondary = A_secondary.astype(float)
        if 'binary' in type_of_adjacency_matrix and 'and' in type_of_adjacency_matrix:
            A_primary = A_primary.astype(float)

    else:
        # Load the data
        data = pd.read_pickle(file_to_load_pkl)
        adj_matrices_df = []  # List to store DataFrames of adjacency matrices
        # Split the string based on "_and_" to handle combinations
        matrix_types = type_of_adjacency_matrix.split("_and_")

        for matrix_type in matrix_types:
            if matrix_type == "binary":
                df_A = generate_A_binary(target_feature)
                # save d_a to parquet
                if 'strava' in target_feature:
                    df_A.to_parquet(BERLIN_ADJACENCY_BINARY)
                if 'taxi' in target_feature:
                    df_A.to_parquet(NY_ADJACENCY_BINARY)
            elif matrix_type == "distance_birdfly":
                df_A = generate_A(feature_selection, data, target_feature)
                if 'strava' in target_feature:
                    df_A.to_parquet(BERLIN_ADJACENCY_BIRD)
                if 'taxi' in target_feature:
                    df_A.to_parquet(NY_ADJACENCY_BIRD)
            elif matrix_type == "distance_traveltime":
                df_A = generate_A_roaddistance_and_roadtime(
                    feature_selection, data, "distance_traveltime", target_feature
                )
                if 'strava' in target_feature:
                    df_A.to_parquet(BERLIN_ADJACENCY_TRAVELTIME)
                if 'taxi' in target_feature:
                    df_A.to_parquet(NY_ADJACENCY_TRAVELTIME)
            elif matrix_type == "distance_roaddistance":
                df_A = generate_A_roaddistance_and_roadtime(
                    feature_selection, data, "distance_roaddistance", target_feature
                )
                if 'strava' in target_feature:
                    df_A.to_parquet(BERLIN_ADJACENCY_ROADDISTANCE)
                if 'taxi' in target_feature:
                    df_A.to_parquet(NY_ADJACENCY_ROADDISTANCE)
            elif matrix_type == "similarity":
                df_A = generate_A_similarity(feature_selection, data, target_feature)
                if 'strava' in target_feature:
                    df_A.to_parquet(BERLIN_ADJACENCY_SIMILARITY)
                if 'taxi' in target_feature:
                    df_A.to_parquet(NY_ADJACENCY_SIMILARITY)
            else:
                raise ValueError(f"Unknown adjacency matrix type: {matrix_type}")
            # Append the DataFrame to the list
            adj_matrices_df.append(df_A)

        # Combine adjacency matrices (example: element-wise average)
        # Use the first counter_ids (assuming all counter_ids are identical)
        if len(adj_matrices_df) > 1:
            A_primary_df = adj_matrices_df[0]
            A_secondary_df = adj_matrices_df[1]
        if len(adj_matrices_df) == 1:
            A_primary_df = adj_matrices_df[0]
            A_secondary_df = adj_matrices_df[0]
        counter_ids = A_primary_df.columns.tolist()
        A_primary = A_primary_df.to_numpy()
        A_secondary = A_secondary_df.to_numpy()

    # if include_entire_graph_test_segments:
    A_primary_training = (
        A_primary.copy()
    )  # include the test sensor as masked element in the train data(so we simply transfer the full adjacency matrix)
    A_primary_validation = A_primary.copy()
    A_secondary_training = A_secondary.copy()
    A_secondary_validation = A_secondary.copy()

    A_primary_test = A_primary
    A_secondary_test = A_secondary

    return (
        A_primary_training,
        A_primary_validation,
        A_primary_test,
        A_secondary_training,
        A_secondary_validation,
        A_secondary_test,
        counter_ids,
    )


def obtain_A_q_and_A_h_and_edges(A, device):
    # logging.info("Obtaining A_q and A_h matrices...")
    A_q = torch.from_numpy(calculate_random_walk_matrix(A).T.astype("float32")).to(
        device
    )
    A_h = torch.from_numpy(calculate_random_walk_matrix(A.T).T.astype("float32")).to(
        device
    )
    # logging.info("A_q and A_h matrices obtained.")
    return A_q, A_h


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    This function is used to compute the distance between sensors.
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        raise ValueError("Longitude values must be between -180 and 180 degrees.")
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        raise ValueError("Latitude values must be between -90 and 90 degrees.")
    dlon = lon2 - lon1
    dlat = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    if c * r * 1000 >= 70000:
        raise ValueError(
            "Computed distances between sensors exceed 70km, which is unrealistic within Berlin."
        )
    return c * r * 1000


def read_relevant_data_for_generate_A(target_feature):
    if "strava" in target_feature:
        data_info = pd.read_parquet(BERLIN_GRAPH)
    elif "taxi" in target_feature:
        data_info = pd.read_parquet(NEW_YORK_GRAPH)
        # keep only the columns counter_name latitude and longitude, edge_index, street_name
        # create variable edge_index
    data_info["edge_index"] = (
        data_info["counter_name"].str.extract(r"(\d+)").astype(int)
    )
    data_info = data_info[["counter_name", "latitude", "longitude", "edge_index"]]

    return data_info


def generate_A(feature_selection, data, target_feature):
    logging.info(
        f"Generating adjacency matrix A for feature_selection={feature_selection} and target_feature={target_feature}..."
    )
    """
    Creates and returns the A (association matrix) of the counting stations as well as an alphabetical list of the counting stations.
    """

    # If the word features slection contains the word 'limited', we only want the data for 2019.
    if "strava" in target_feature and "limited" in feature_selection:
        data = data[data["date"].str.contains("2019")]

    elif "taxi" in target_feature and "limited" in feature_selection:
        data = data[data["date"].str.contains("2016-01")]

    elif "taxi" in target_feature and "full" in feature_selection:
        data = data[data["date"].str.contains("2016-01|2016-02", regex=True)]

    # read the relevant data
    data_info = read_relevant_data_for_generate_A(target_feature)

    # We drop the superfluous counter_name which are not in data['counter_name'].
    data_info = data_info[data_info["counter_name"].isin(data["counter_name"])]

    # Make sure the data is in the same order as counter_ids.
    data_info = data_info.sort_values(by="counter_name")
    data_info = data_info.reset_index(drop=True)

    # Get the adjacency matrix
    counter_ids = data_info["counter_name"].unique()
    n_counters = len(counter_ids)
    A = np.zeros((len(counter_ids), len(counter_ids)))
    assert data_info["counter_name"].values.tolist() == counter_ids.tolist()

    # Compute distance between sensors.
    for i, row in enumerate(data_info.iterrows()):
        print(f"Computing distance for sensor {i+1}/{n_counters}")
        for j in range(i, n_counters):
            row2 = data_info.iloc[j]
            lng1 = row[1]["longitude"]
            lng2 = row2["longitude"]
            lat1 = row[1]["latitude"]
            lat2 = row2["latitude"]
            d = haversine(lng1, lat1, lng2, lat2)
            A[i, j] = d
            if i != j:
                A[j, i] = d

    A[i, j] = d**2
    A = A / 7500  # distance / 7.5 km
    A = np.exp(-A)
    logging.info("Adjacency matrix A generated.")

    A = pd.DataFrame(A, index=counter_ids, columns=counter_ids)

    return A


def generate_A_similarity(feature_selection, data, target_feature):
    """
    Creates and returns the A (association matrix) of the counting stations as well as an alphabetical list of the counting stations.
    """

    # If the word features slection contains the word 'limited', we only want the data for 2019.
    if "strava" in target_feature and "limited" in feature_selection:
        data = data[data["date"].str.contains("2019")]

    elif "taxi" in target_feature and "limited" in feature_selection:
        data = data[data["date"].str.contains("2016-01")]

    elif "taxi" in target_feature and "full" in feature_selection:
        data = data[data["date"].str.contains("2016-01|2016-02", regex=True)]

    # read the relevant data
    data_info = read_relevant_data_for_generate_A(target_feature)

    # We drop the superfluous counter_name which are not in data['counter_name'].
    data_info = data_info[data_info["counter_name"].isin(data["counter_name"])]

    # Make sure the data is in the same order as counter_ids.
    data_info = data_info.sort_values(by="counter_name")
    data_info = data_info.reset_index(drop=True)

    # Get the adjacency matrix
    counter_ids = data_info["counter_name"].unique()
    n_counters = len(counter_ids)
    A = np.zeros((len(counter_ids), len(counter_ids)))
    assert data_info["counter_name"].values.tolist() == counter_ids.tolist()

    # Get all columsn which start with infrastructure_
    infrastructure_columns = [
        column_name
        for column_name in data.columns
        if column_name.startswith("infrastructure_")
    ]

    ## Assert that these columns only vary across space (so betewen counter_name, but not across dates)
    for column_name in infrastructure_columns:
        max_unique_values = data.groupby("counter_name")[column_name].nunique().max()
        assert (
            max_unique_values == 1
        ), f"Column '{column_name}' has more than one unique value per 'counter_name'. Max unique values: {max_unique_values}. This means, the column varies across dates, which is not allowed (not time invaariant)."

    # Filter dataset to one random date
    data = data[data["date"] == data["date"].iloc[0]]

    # Drop unnecessary columns
    data = data.drop(
        columns=[
            col
            for col in data.columns
            if col not in infrastructure_columns + ["counter_name"]
        ]
    )

    # Identify categorical and numerical columns
    categorical_cols = (
        data[infrastructure_columns]
        .select_dtypes(include=["object", "category"])
        .columns
    )
    numerical_cols = list(set(infrastructure_columns) - set(categorical_cols))

    # One-hot encode categorical features
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Normalize numerical features (optional but recommended)
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Set counter_name as index for fast lookup
    data = data.set_index("counter_name")

    # Ensure counters are sorted correctly
    data_info = data_info.sort_values(by="counter_name").reset_index(drop=True)
    counter_ids = data_info["counter_name"].values
    assert data_info["counter_name"].values.tolist() == counter_ids.tolist()

    # Extract infrastructure feature matrix
    X_infra = data.loc[counter_ids].values  # Ensure the correct order

    # Compute cosine similarity matrix
    A = cosine_similarity(X_infra)

    A = pd.DataFrame(A, index=counter_ids, columns=counter_ids)

    return A


def generate_A_roaddistance_and_roadtime(
    feature_selection, data, type_of_travel_distance, target_feature
):
    """
    Creates and returns the A (association matrix) of the counting stations as well as an alphabetical list of the counting stations.
    """
    if "strava" in target_feature and "limited" in feature_selection:
        data = data[data["date"].str.contains("2019")]
    elif "taxi" in target_feature and "limited" in feature_selection:
        data = data[data["date"].str.contains("2016-01")]
    elif "taxi" in target_feature and "full" in feature_selection:
        data = data[data["date"].str.contains("2016-01|2016-02", regex=True)]

    # read the relevant data
    data_info = read_relevant_data_for_generate_A(target_feature)

    # We drop the superfluous counter_name which are not in data['counter_name'].
    data_info = data_info[data_info["counter_name"].isin(data["counter_name"])]

    # Make sure the data is in the same order as counter_ids.
    data_info = data_info.sort_values(by="counter_name")
    data_info = data_info.reset_index(drop=True)

    # We have the travel time and travel distance precomputed.
    # Load them
    if type_of_travel_distance == "distance_traveltime" and "strava" in target_feature:
        data_travel = pd.read_pickle(BERLIN_MATRIX_TRAVELTIME)
    elif (
        type_of_travel_distance == "distance_roaddistance"
        and "strava" in target_feature
    ):
        data_travel = pd.read_pickle(BERLIN_MATRIX_TRAVELDISTANCE)
    elif type_of_travel_distance == "distance_traveltime" and "taxi" in target_feature:
        data_travel = pd.read_pickle(NY_MATRIX_TRAVELTIME)
    elif (
        type_of_travel_distance == "distance_roaddistance" and "taxi" in target_feature
    ):
        data_travel = pd.read_pickle(NY_MATRIX_TRAVELDISTANCE)
    else:
        raise ValueError(
            "type_of_travel_distance must be either 'distance_traveltime' or 'distance_roaddistance'"
        )

    # # Only keep those columns / rows in data_travel that are in data_info
    data_travel = data_travel[data_info["counter_name"]]
    data_travel = data_travel.loc[data_info["counter_name"]]
    counter_ids = np.array(data_travel.columns.tolist())
    # convert to array
    counter_ids = np.array(counter_ids)

    # Turn into numpy array
    data_travel = data_travel.to_numpy()

    if type_of_travel_distance == "distance_traveltime":
        if target_feature == "strava_total_trip_count":
            # We want to have the travel time, not the travel distance
            sigma = 6000
        elif target_feature == "taxi_total_trip_count":
            sigma = 10000
    elif type_of_travel_distance == "distance_roaddistance":
        # We want to have the travel distance, not the travel time
        if target_feature == "strava_total_trip_count":
            sigma = 3000
        elif target_feature == "taxi_total_trip_count":
            sigma = 850

    exponent = 2

    # See function in paper
    distance_matrix = data_travel / sigma  # Element-wise division by sigma
    powered_distance_matrix = np.power(
        distance_matrix, exponent
    )  # Squaring the result of division
    A = np.exp(-powered_distance_matrix)  # Apply the exponential function

    A = pd.DataFrame(A, index=counter_ids, columns=counter_ids)

    return A


def generate_A_binary(target_feature):
    if "strava" in target_feature:
        data_info = pd.read_parquet(BERLIN_GRAPH)
    elif "taxi" in target_feature:
        data_info = pd.read_parquet(NEW_YORK_GRAPH)
        # keep only the columns counter_name latitude and longitude, edge_index, street_name
        # create variable edge_index
    data_info["edge_index"] = (
        data_info["counter_name"].str.extract(r"(\d+)").astype(int)
    )
    data_info = data_info[["counter_name", "latitude", "longitude", "edge_index"]]
    # Load the actual graph data, based on which we will compute the connectivity measures
    if "strava" in target_feature:
        with open(PROCESSED_BERLIN_CYCLING_NETWORK_GRAPH, "rb") as f:
            G = pkl.load(f)
    elif "taxi" in target_feature:
        with open(NEW_YORK_INVERTED_UNDIRECTED_GRAPH, "rb") as f:
            G = pkl.load(f)
    assert len(G.nodes) == len(data_info)
    # Create a binary adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()
    binary_adj_matrix = (adj_matrix != 0).astype(int)
    counter_names = data_info["counter_name"]
    # I want them sorted alphabetiacally, to correspond with the remaining code.
    sorted_indices = np.argsort(
        counter_names
    )  # Step 1: Sort the counter names alphabetically
    sorted_counter_names = np.array(counter_names)[sorted_indices]
    sorted_binary_adj_matrix = binary_adj_matrix[sorted_indices, :][
        :, sorted_indices
    ]  # Step 2: Reorder the binary adjacency matrix based on the sorted counter names
    # Set the diagonal to 1s
    np.fill_diagonal(sorted_binary_adj_matrix, 1)

    sorted_binary_adj_matrix = pd.DataFrame(
        sorted_binary_adj_matrix,
        index=sorted_counter_names,
        columns=sorted_counter_names,
    )

    return sorted_binary_adj_matrix  # all in alphabetical order
