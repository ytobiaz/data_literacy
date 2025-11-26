import os
import pickle as pkl
import numpy as np
import pandas as pd
import random
import torch
from typing import Tuple, Set
from h_adjacency_matrices import (
    load_adjacency_matrix_and_counter_ids,
    obtain_A_q_and_A_h_and_edges,
)

import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import logging
from paths import (
    WORKING_BERLIN_DATA,
    WORKING_NEWYORK_DATA,
    FEATURE_LIST_STRAVA,
    FEATURE_LIST_NEWYORK,
    GNNUI_DATA_FACSIMILE_STORAGE,
    GNNUI_DATA_STRAVA,
    GNNUI_DATA_TAXI,
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_data_for_GNNUI(feature_selection):
    logging.info(
        f"Creating data for GNNUI with feature_selection={feature_selection}..."
    )
    if feature_selection in ["full_data_all_features_for_strava_as_target", "all_features_for_strava_as_target"]:
        data = pd.read_csv(WORKING_BERLIN_DATA)
        file_to_load = FEATURE_LIST_STRAVA  # BLD_FEAUTRE_LIST / "handpicked_features_for_strava_as_target.json"
    #     GNNUI / "working_data_minmax_for_gnnui_facsimile_for_strava_as_target.pkl"
    # )
    elif feature_selection in ["all_features_for_taxi_as_target", "full_data_all_features_for_taxi_as_target"]:
        data = pd.read_csv(WORKING_NEWYORK_DATA)
        file_to_load = FEATURE_LIST_NEWYORK  # BLD_FEATURE_LIST_HANDPICKED_NEWYORK = (
    else:
        raise ValueError(
            "feature_selection should be either 'all_features_for_taxi_as_target' or 'all_features_for_strava_as_target'"
        )
    with open(file_to_load, "r") as file:
        feature_list = json.load(file)
        target = feature_list["target"]
        numerical = feature_list["numerical"]
        binary = feature_list["binary"]
        categorical = feature_list["categorical"]

    # Save the counter_column
    counters = data["counter_name"]

    # check if all the columns are in the data, if not remove them totally
    binary = [col for col in binary if col in data.columns]
    numerical = [col for col in numerical if col in data.columns]
    categorical = [col for col in categorical if col in data.columns]

    # only keep those columns
    columns_to_keep = target + numerical + binary + categorical + ["date"]
    data = data[columns_to_keep]

    # Merge counter_name back
    data["counter_name"] = counters

    # assert counter_name and date in data
    assert "counter_name" in data.columns
    assert "date" in data.columns

    # assert the data is balanced
    assert data["counter_name"].nunique() * data["date"].nunique() == len(data)

    # for Strava, we drop the outliers, (more than 3 standard deviations)
    if "for_strava_as_target" in feature_selection:
        column_name = "strava_total_trip_count"
    elif "for_taxi_as_target" in feature_selection:
        column_name = "taxi_total_trip_count"

    group_column = "counter_name"

    # Calculate mean and std_dev per group
    grouped = data.groupby(group_column)
    data["z_score"] = grouped[column_name].transform(lambda x: (x - x.mean()) / x.std())

    # Define threshold for z-score
    threshold = 3

    # Set outliers to NaN for values outside the threshold
    data.loc[abs(data["z_score"]) > threshold, column_name] = np.nan

    # Drop the z_score column
    data = data.drop(columns=["z_score"])
    if "taxi" in feature_selection:
        data["date"] = data["date"].astype(str)

    # Save the data
    if "for_strava_as_target" in feature_selection:
        data.to_pickle(GNNUI_DATA_STRAVA)
    elif "for_taxi_as_target" in feature_selection:
        data.to_pickle(GNNUI_DATA_TAXI)


def generate_training_and_test_set(
    data,
    feature_selection,
    counter_ids,
    test_nodes_set,
    only_extrapolation_and_no_forecasting,
    percentage_of_the_training_data_used_for_validation,
    standardize_columns=True,
    target_feature_input="count",
):
    logging.info("Generating training and test sets...")
    """
    Create and return the test set and training set for the berlin cycling sensor data.
    """

    # Only keep data for 2019 (This is the case for limited data. For the full data, we keep all years (2019-2023))
    if "limited" in feature_selection and "strava" in target_feature_input:
        data = data[data["date"].str.contains("2019")]
    elif "limited" in feature_selection and "taxi" in target_feature_input:
        # only keep jan 2016, date is datetime object
        data = data[data["date"].str.contains("2016-01")]
    elif "full" in feature_selection and "taxi" in target_feature_input:
        # only keep jan and feb 2016, date is datetime object
        data = data[data["date"].str.contains("2016-01|2016-02", regex=True)]

    # Use .loc to set the new column
    data.loc[:, "counter_names"] = data["counter_name"]

    # Get the numerical , binary and categorical columns
    binary_columns = data.select_dtypes(include=[bool]).columns.tolist()
    categorical_columns = data.select_dtypes(include=[object]).columns.tolist()
    numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    list_of_columns_non_changed = ["date", "counter_name", target_feature_input]
    # remove target feature from numerical columns
    numerical_columns.remove(target_feature_input)
    categorical_columns.remove("counter_name")
    categorical_columns.remove("date")

    # Split the data into train and test data (at 70%) and standardize and one hot encode the data
    data = data.sort_values(by="date")
    if only_extrapolation_and_no_forecasting:
        training_set = data.copy()
        validation_set = data.copy()
        test_set = data.copy()
    else:
        split_date_train = data["date"].unique()[int(len(data["date"].unique()) * 0.7)]
        split_date_validation = data["date"].unique()[
            int(len(data["date"].unique()) * 0.85)
        ]
        # assign those observations up until split_date to the training set and the rest to the test set
        training_set = data[data["date"] <= split_date_train]
        validation_set = data[
            (data["date"] > split_date_train) & (data["date"] <= split_date_validation)
        ]
        test_set = data[data["date"] > split_date_validation]

    # Choose the nodes, which are in the validation set
    test_nodes_set = set(test_nodes_set)
    all_nodes_set = set(
        range(0, len(counter_ids))
    )  # how many locations are there (e.g. 22 counting stations)
    all_nodes_set_length = len(counter_ids)
    opposite_test_set = all_nodes_set - test_nodes_set
    # pick 0.9 of the opposite set as the training set and the remaining 10% as the validation set
    validation_nodes_set = set(
        np.random.choice(
            list(opposite_test_set),
            int(
                len(opposite_test_set)
                * (percentage_of_the_training_data_used_for_validation)
            ),
            replace=False,
        )
    )  # we take 10% of the train data as validation, possibly change this?
    train_nodes_set = all_nodes_set - test_nodes_set - validation_nodes_set
    # assert no same nodes in any of the sets
    assert len(train_nodes_set.intersection(validation_nodes_set)) == 0
    assert len(train_nodes_set.intersection(test_nodes_set)) == 0
    assert len(validation_nodes_set.intersection(test_nodes_set)) == 0
    assert len(train_nodes_set) + len(validation_nodes_set) + len(
        test_nodes_set
    ) == len(all_nodes_set)

    # drop the unknown set from the training set
    # Get the name of the counting stations, which corresponds to the unknown_set (integer) of counter_ids
    test_counting_station_names = [counter_ids[i] for i in test_nodes_set]
    validation_counting_station_names = [counter_ids[i] for i in validation_nodes_set]

    # What is a nan value in our data
    nan_value = -1

    # if include_entire_graph_test_segments:
    # replace all the count values of the unknwon set with nan
    training_set.loc[
        training_set["counter_name"].isin(
            test_counting_station_names + validation_counting_station_names
        ),
        target_feature_input,
    ] = nan_value
    validation_set.loc[
        validation_set["counter_name"].isin(test_counting_station_names),
        target_feature_input,
    ] = nan_value

    # if the target feature is strava, we need to remove the counter_name from the numerical columns
    categorical_columns.remove("counter_names")

    # Standardize the data
    # bring the columns in the right order (so that the columns are right after teh pipeline
    # bring them in the order numerical, categorical, binary, non-changed
    training_set = training_set.reindex(
        columns=numerical_columns
        + categorical_columns
        + binary_columns
        + list_of_columns_non_changed
    )
    validation_set = validation_set.reindex(
        columns=numerical_columns
        + categorical_columns
        + binary_columns
        + list_of_columns_non_changed
    )
    test_set = test_set.reindex(
        columns=numerical_columns
        + categorical_columns
        + binary_columns
        + list_of_columns_non_changed
    )
    if standardize_columns:
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", MinMaxScaler(feature_range=(-1, 1)), numerical_columns),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_columns + binary_columns,
                ),
            ],
            remainder="passthrough",  # Include remaining columns
        )
    if not standardize_columns:
        preprocessor = ColumnTransformer(
            # do nothing to the numeric columns
            transformers=[
                ("numeric", "passthrough", numerical_columns),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_columns + binary_columns,
                ),
            ],
            remainder="passthrough",  # Include remaining columns
        )
    pipeline = Pipeline([("preprocessor", preprocessor)])
    transformed_training_set = pipeline.fit_transform(training_set)
    transformed_validation_set = pipeline.transform(validation_set)
    transformed_test_set = pipeline.transform(test_set)
    # Get the column names for the transformed data
    transformed_column_names = (
        numerical_columns
        + list(
            pipeline.named_steps["preprocessor"]
            .named_transformers_["categorical"]
            .get_feature_names_out(categorical_columns + binary_columns)
        )
        + list_of_columns_non_changed  # Include non-transformed columns
    )
    # Convert the transformed arrays back to DataFrames
    training_set = pd.DataFrame(
        transformed_training_set, columns=transformed_column_names
    )
    validation_set = pd.DataFrame(
        transformed_validation_set, columns=transformed_column_names
    )
    test_set = pd.DataFrame(transformed_test_set, columns=transformed_column_names)

    # Get a list of the features, to assure that we append it always in the same order.
    features = training_set.columns.tolist()
    features.remove("date")
    features.remove("counter_name")
    features.remove(target_feature_input)

    # Obtain the counter_ids minus the unknown set
    counter_ids_training = [  # contain all counter ids in the training set
        counter
        for counter in counter_ids
        if counter
        not in test_counting_station_names + validation_counting_station_names
    ]
    assert len(counter_ids_training) == len(train_nodes_set)
    counter_ids_validation = [  # contain all counter ids in the validation set (so those in train and those exclusively in validation)
        counter for counter in counter_ids if counter not in test_counting_station_names
    ]
    assert len(counter_ids_validation) == len(train_nodes_set) + len(
        validation_nodes_set
    )
    # Get the X (consisting of the dependent and independent features).

    for data_set in ["train", "validation", "test"]:

        if data_set == "train":
            data = training_set
        elif data_set == "validation":
            data = validation_set
        else:
            data = test_set
        X = np.zeros((len(counter_ids), len(data["date"].unique()), len(features) + 1))
        counter_ids_to_iterate = counter_ids.copy()
        #  Sort
        data["counter_name"] = pd.Categorical(
            data["counter_name"], categories=counter_ids_to_iterate, ordered=True
        )
        data = data.sort_values(by=["counter_name", "date"])
        # drop date and counter_name
        data = data.drop(columns=["date"])
        # Add the count value first (we want them in dimension 0)

        # Set the nan values in count to zero (.fillna(0))
        for i, counter in enumerate(counter_ids_to_iterate):
            data_temp = data[data["counter_name"] == counter]
            data_temp = data_temp.drop(columns=["counter_name"])
            count_values = data_temp[target_feature_input].values
            data_temp = data_temp.drop(columns=[target_feature_input])
            X[i, : len(count_values), 0] = count_values
            feature_values = data_temp.values
            X[i, : len(count_values), 1 : 1 + len(features)] = feature_values
        nan_mask = np.isnan(X[:, :, 0])
        X[nan_mask, 0] = nan_value  # no zero counts in ground truth

        if data_set == "train":
            training_set_X = X.copy()
        elif data_set == "validation":
            validation_set_X = X.copy()
        else:
            test_set_X = X.copy()

    if only_extrapolation_and_no_forecasting:
        assert training_set_X.shape == validation_set_X.shape == test_set_X.shape
        assert (training_set_X[:, :, 1:] == test_set_X[:, :, 1:]).all()
        assert (training_set_X[:, :, 1:] == validation_set_X[:, :, 1:]).all()
        assert (
            test_set_X[list(validation_nodes_set), :, :]
            == validation_set_X[list(validation_nodes_set), :, :]
        ).all()
        assert (
            test_set_X[list(train_nodes_set), :, :]
            == validation_set_X[list(train_nodes_set), :, :]
        ).all()
        assert (
            training_set_X[list(train_nodes_set), :, :]
            == validation_set_X[list(train_nodes_set), :, :]
        ).all()
        assert not (training_set_X[:, :, :] == test_set_X[:, :, :]).all()
        assert not (training_set_X[:, :, :] == validation_set_X[:, :, :]).all()
    else:
        assert training_set_X.shape[1] > test_set_X.shape[1]
        assert training_set_X.shape[1] > validation_set_X.shape[1]
        assert (
            training_set_X.shape[0] == test_set_X.shape[0] == validation_set_X.shape[0]
        )
        assert (
            training_set_X.shape[2] == test_set_X.shape[2] == validation_set_X.shape[2]
        )
    assert training_set_X.shape[0] == all_nodes_set_length
    assert training_set_X.shape[2] == len(features) + 1
    logging.info("Training and test sets generated.")
    return (
        training_set_X,
        validation_set_X,
        test_set_X,
        train_nodes_set,
        validation_nodes_set,
        all_nodes_set,
    )  # , validation_nodes_set_new, train_nodes_set_new, test_nodes_set_new


def get_or_load_data(
    test_nodes_set,
    feature_selection,
    only_extrapolation_and_no_forecasting,
    type_of_adjacency_matrix,
    percentage_of_the_training_data_used_for_validation,
    target_feature,
    percentage_of_segments_used_for_test,
    cross_validation_group,
):
    """
    Loads or generates data for training, validation, and testing.

    If a precomputed dataset exists, it loads the data; otherwise, it generates and saves it.
    Returns:
        tuple: Contains adjacency matrices, datasets, and node information.
    """
    file_path_template = (
        GNNUI_DATA_FACSIMILE_STORAGE
        / (  # GNNUI_DATA_FACSIMILE_STORAGE = GNNUI / "data_facsimile_prepared"
            f"index_{test_nodes_set[0]}"
            f"_featuresel{feature_selection}"
            f"_extrapol{only_extrapolation_and_no_forecasting}"
            f"_targetfeature{target_feature}"
            f"_percentageoftrain{percentage_of_the_training_data_used_for_validation}"
            f"_percentageusedfortesting{percentage_of_segments_used_for_test}"
        )
    )

    file_path = f"{file_path_template}.pkl"
    (
        A_training_primary,
        A_validation_primary,
        A_test_primary,
        A_training_secondary,
        A_validation_secondary,
        A_test_secondary,
        counting_station_names,
    ) = load_adjacency_matrix_and_counter_ids(
        feature_selection, type_of_adjacency_matrix, target_feature
    )

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            (
                training_set_X,
                validation_set_X,
                test_set_X,
                counter_ids,
                test_nodes_set,
                all_nodes_set,
                train_nodes_set,
                validation_nodes_set,
            ) = pkl.load(f)
    else:
        (
            training_set_X,
            validation_set_X,
            test_set_X,
            counter_ids,
            test_nodes_set,
            all_nodes_set,
            train_nodes_set,
            validation_nodes_set,
        ) = load_data(
            test_nodes_set,
            feature_selection,
            only_extrapolation_and_no_forecasting,
            percentage_of_the_training_data_used_for_validation,
            target_feature,
            counting_station_names,
        )
        if (
            cross_validation_group == 0
        ):  # we do not save for others (too much space on device)
            with open(file_path, "wb") as f:
                pkl.dump(
                    (
                        training_set_X,
                        validation_set_X,
                        test_set_X,
                        counting_station_names,
                        test_nodes_set,
                        all_nodes_set,
                        train_nodes_set,
                        validation_nodes_set,
                    ),
                    f,
                )

    assert np.array_equal(
        counter_ids, counter_ids
    ), "The counting station names do not match the counter ids."

    return (
        A_training_primary,
        A_validation_primary,
        A_test_primary,
        A_training_secondary,
        A_validation_secondary,
        A_test_secondary,
        training_set_X,
        validation_set_X,
        test_set_X,
        counting_station_names,
        test_nodes_set,
        all_nodes_set,
        train_nodes_set,
        validation_nodes_set,
    )


def adjust_node_sets(nodes_set: set, percentage: float, total_size: float) -> set:
    """
    Select a subset of nodes based on the given percentage.

    Parameters:
    - nodes_set: The set of nodes to select from.
    - percentage: The percentage of nodes to select.
    - total_size: The total size of the dataset.

    Returns:
    - A subset of nodes as a set.
    """
    num_nodes = int(total_size * percentage)
    ordered_nodes = random.sample(list(nodes_set), len(nodes_set))
    return set(ordered_nodes[:num_nodes])


def mask_or_remove_data(
    dataset: np.ndarray, nodes_to_mask: set, mask_value: float = np.nan
) -> np.ndarray:
    """
    Mask or remove data for the specified nodes in the dataset.

    Parameters:
    - dataset: The dataset to modify.
    - nodes_to_mask: A set of node indices to mask or remove.
    - mask_value: The value to use for masking. Default is np.nan.

    Returns:
    - Updated dataset with specified nodes masked or removed.
    """
    dataset[:, list(nodes_to_mask), 0] = mask_value
    return dataset


def update_adjacency_matrices(
    adj_matrix: np.ndarray, nodes_to_remove: set
) -> np.ndarray:
    """
    Remove rows and columns corresponding to nodes_to_remove from the adjacency matrix.

    Parameters:
    - adj_matrix: The adjacency matrix to update.
    - nodes_to_remove: A set of node indices to remove.

    Returns:
    - Updated adjacency matrix with specified nodes removed.
    """
    adj_matrix = np.delete(adj_matrix, list(nodes_to_remove), axis=0)  # Remove rows
    adj_matrix = np.delete(adj_matrix, list(nodes_to_remove), axis=1)  # Remove columns
    return adj_matrix


def employ_the_right_percentage_of_training_data_and_maybe_exclude_test(
    include_entire_graph_train_segments: bool,
    include_entire_graph_test_segments: bool,
    test_set: np.ndarray,
    validation_set: np.ndarray,
    training_set: np.ndarray,
    train_nodes_set: Set[int],
    validation_nodes_set: Set[int],
    test_nodes_set: Set[int],
    all_nodes_set: Set[int],
    A_test_primary: np.ndarray,
    A_val_primary: np.ndarray,
    A_train_primary: np.ndarray,
    A_test_secondary: np.ndarray,
    A_val_secondary: np.ndarray,
    A_train_secondary: np.ndarray,
    percentage_of_segments_used_for_training: float,
    percentage_of_segments_used_for_test: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Set[int],
    Set[int],
    Set[int],
    Set[int],
    Set[int],
]:
    """
    Adjust the training, validation, and test datasets based on the specified percentages.

    Parameters:
    - include_entire_graph_train_segments: Whether to include hidden training data.
    - include_entire_graph_test_segments: Whether to include the test sensor in training data.
    - test_set, validation_set, training_set: Datasets to adjust.
    - train_nodes_set, validation_nodes_set, test_nodes_set: Node sets.
    - all_nodes_set: Set of all nodes.
    - A_test_primary, A_val_primary, A_train_primary: Primary adjacency matrices.
    - A_test_secondary, A_val_secondary, A_train_secondary: Secondary adjacency matrices.
    - percentage_of_segments_used_for_training: Percentage of segments used for training.
    - percentage_of_segments_used_for_test: Percentage of segments used exclusively for testing.

    Returns:
    - Updated datasets, adjacency matrices, and node sets.
    """
    random.seed(42)
    training_set_size_if100 = len(train_nodes_set) / (
        1 - percentage_of_segments_used_for_test
    )
    validation_set_size_if100 = len(validation_nodes_set) / (
        1 - percentage_of_segments_used_for_test
    )

    # Select the first N elements to ensure consistency
    chosen_set_train = adjust_node_sets(
        train_nodes_set,
        percentage_of_segments_used_for_training,
        training_set_size_if100,
    )
    chosen_set_val = adjust_node_sets(
        validation_nodes_set,
        percentage_of_segments_used_for_training,
        validation_set_size_if100,
    )

    train_nodes_proof_of_concept = chosen_set_train.copy()
    validation_nodes_proof_of_concept = chosen_set_val.copy()

    not_chosen_in_train = train_nodes_set - chosen_set_train
    not_chosen_in_val = validation_nodes_set - chosen_set_val
    test_nodes_set = set(test_nodes_set)

    nan_value = -1

    if include_entire_graph_train_segments:

        # In this case, we simply set the hidden ones to zero, in the training and testing data.
        if include_entire_graph_test_segments:
            test_set = mask_or_remove_data(test_set, not_chosen_in_train, nan_value)
            test_set = mask_or_remove_data(test_set, not_chosen_in_val, nan_value)
            validation_set = mask_or_remove_data(
                validation_set, not_chosen_in_train, nan_value
            )
            validation_set = mask_or_remove_data(
                validation_set, not_chosen_in_val, nan_value
            )
            training_set = mask_or_remove_data(
                training_set, not_chosen_in_train, nan_value
            )
            validation_nodes_set_updated = validation_nodes_set.copy()
            train_nodes_set_updated = train_nodes_set.copy()
            test_nodes_set_updated = test_nodes_set.copy()
        else:
            # We begin by setting the test sensors to nan, as this poses no problem with the indexes.
            test_set = mask_or_remove_data(test_set, not_chosen_in_train, np.nan)
            test_set = mask_or_remove_data(test_set, not_chosen_in_val, np.nan)
            validation_set = mask_or_remove_data(
                validation_set, not_chosen_in_train, np.nan
            )
            validation_set = mask_or_remove_data(
                validation_set, not_chosen_in_val, np.nan
            )
            training_set = mask_or_remove_data(
                training_set, not_chosen_in_train, np.nan
            )
            # Next we delete the not_chosen_in_val from the training set, bceause include_entire_graph_test_segments = False
            training_set = np.delete(
                training_set, list(test_nodes_set | validation_nodes_set), axis=1
            )
            validation_set = np.delete(validation_set, list(test_nodes_set), axis=1)

            # We also need to adjust A
            A_train_primary = update_adjacency_matrices(
                A_train_primary, test_nodes_set | validation_nodes_set
            )
            A_train_secondary = update_adjacency_matrices(
                A_train_secondary, test_nodes_set | validation_nodes_set
            )
            A_val_primary = update_adjacency_matrices(A_val_primary, test_nodes_set)
            A_val_secondary = update_adjacency_matrices(A_val_secondary, test_nodes_set)

            # We need updated indexes
            remaining_nodes_train = all_nodes_set - (
                test_nodes_set | validation_nodes_set
            )
            index_mapping_train = {
                old_idx: new_idx
                for new_idx, old_idx in enumerate(sorted(remaining_nodes_train))
            }
            # We did not delete any train nodes
            missing_nodes = [
                node
                for node in sorted(train_nodes_set)
                if node not in index_mapping_train
            ]
            # assert not missing_nodes, f"Some train nodes are missing from index_mapping_train: {missing_nodes}"
            train_nodes_set_updated = [
                index_mapping_train[node]
                for node in sorted(train_nodes_set)
                if node in index_mapping_train
            ]

            # index_mapping_val = {old_idx: new_idx for new_idx, old_idx in enumerate(validation_nodes_set.remove(test_nodes_set))}
            remaining_nodes_val = sorted(all_nodes_set - test_nodes_set)
            index_mapping_val = {
                old_idx: new_idx
                for new_idx, old_idx in enumerate(sorted(remaining_nodes_val))
            }
            # we did not delete any nodes in validation set, we only set them to nan
            # assert len(remaining_nodes_val) == validation_set.shape[1], 'The number of nodes in the validation set does not match the number of nodes in the all_nodes_set.'
            validation_nodes_set_updated = [
                index_mapping_val[node]
                for node in sorted(validation_nodes_set)
                if node in index_mapping_val
            ]

            # test set has the same length as before
            test_nodes_set_updated = test_nodes_set.copy()

    else:

        if include_entire_graph_test_segments:

            # We begin by setting the test sensors to nan, as this poses no problem with the indexes.
            # (include_entire_graph_test_segments = False)
            # Here we now need to drop
            test_set = np.delete(
                test_set, list(not_chosen_in_train | not_chosen_in_val), axis=1
            )

            validation_set = np.delete(
                validation_set, list(not_chosen_in_train | not_chosen_in_val), axis=1
            )

            training_set = np.delete(
                training_set, list(not_chosen_in_train | not_chosen_in_val), axis=1
            )

            # also make A smaller
            A_test_primary = update_adjacency_matrices(
                A_test_primary, not_chosen_in_train | not_chosen_in_val
            )
            A_test_secondary = update_adjacency_matrices(
                A_test_secondary, not_chosen_in_train | not_chosen_in_val
            )
            A_val_primary = update_adjacency_matrices(
                A_val_primary, not_chosen_in_train | not_chosen_in_val
            )
            A_val_secondary = update_adjacency_matrices(
                A_val_secondary, not_chosen_in_train | not_chosen_in_val
            )
            A_train_primary = update_adjacency_matrices(
                A_train_primary, not_chosen_in_train | not_chosen_in_val
            )
            A_train_secondary = update_adjacency_matrices(
                A_train_secondary, not_chosen_in_train | not_chosen_in_val
            )

            # assert they all have the same shape now
            assert (
                training_set.shape[1] == validation_set.shape[1]
            ), "The training and validation set do not have the same shape."
            assert (
                training_set.shape[1] == test_set.shape[1]
            ), "The training and test set do not have the same shape."
            assert (
                A_train_primary.shape == A_val_primary.shape
            ), "The training and validation set do not have the same shape."
            assert (
                A_train_primary.shape == A_test_primary.shape
            ), "The training and test set do not have the same shape."

            # We need updated indexes
            remaining_nodes = all_nodes_set - (not_chosen_in_train | not_chosen_in_val)
            index_mapping = {
                old_idx: new_idx
                for new_idx, old_idx in enumerate(sorted(remaining_nodes))
            }
            # Validation
            validation_nodes_set_temp = chosen_set_val
            assert (
                len(remaining_nodes) == validation_set.shape[1]
            ), "The number of nodes in the validation set does not match the number of nodes in the all_nodes_set."
            missing_nodes = [
                node for node in validation_nodes_set_temp if node not in index_mapping
            ]
            assert (
                not missing_nodes
            ), f"Some train nodes are missing from index_mapping_train: {missing_nodes}"
            validation_nodes_set_updated = [
                index_mapping[node]
                for node in sorted(validation_nodes_set_temp)
                if node in index_mapping
            ]
            # Test
            test_nodes_set_temp = test_nodes_set.copy()
            assert (
                len(remaining_nodes) == test_set.shape[1]
            ), "The number of nodes in the test set does not match the number of nodes in the all_nodes_set."
            missing_nodes = [
                node for node in test_nodes_set_temp if node not in index_mapping
            ]
            assert (
                not missing_nodes
            ), f"Some train nodes are missing from index_mapping_train: {missing_nodes}"
            test_nodes_set_updated = [
                index_mapping[node]
                for node in sorted(test_nodes_set_temp)
                if node in index_mapping
            ]
            # Train
            train_nodes_set_temp = chosen_set_train
            assert (
                len(remaining_nodes) == training_set.shape[1]
            ), "The number of nodes in the training set does not match the number of nodes in the all_nodes_set."
            missing_nodes = [
                node for node in train_nodes_set_temp if node not in index_mapping
            ]
            assert (
                not missing_nodes
            ), f"Some train nodes are missing from index_mapping_train: {missing_nodes}"
            train_nodes_set_updated = [
                index_mapping[node]
                for node in sorted(train_nodes_set_temp)
                if node in index_mapping
            ]

        else:
            # Here we now need to drop a lot! Thus, we do it jointly, to not mix up the indexes.

            training_set = np.delete(
                training_set,
                list(
                    test_nodes_set
                    | validation_nodes_set
                    | not_chosen_in_train
                    | not_chosen_in_val
                ),
                axis=1,
            )
            validation_set = np.delete(
                validation_set,
                list(test_nodes_set | not_chosen_in_train | not_chosen_in_val),
                axis=1,
            )
            test_set = np.delete(
                test_set, list(not_chosen_in_train | not_chosen_in_val), axis=1
            )

            # also make A smaller
            A_test_primary = update_adjacency_matrices(
                A_test_primary, not_chosen_in_train | not_chosen_in_val
            )
            A_test_secondary = update_adjacency_matrices(
                A_test_secondary, not_chosen_in_train | not_chosen_in_val
            )

            A_val_primary = update_adjacency_matrices(
                A_val_primary, test_nodes_set | not_chosen_in_train | not_chosen_in_val
            )
            A_val_secondary = update_adjacency_matrices(
                A_val_secondary,
                test_nodes_set | not_chosen_in_train | not_chosen_in_val,
            )

            A_train_primary = update_adjacency_matrices(
                A_train_primary,
                test_nodes_set
                | validation_nodes_set
                | not_chosen_in_train
                | not_chosen_in_val,
            )
            A_train_secondary = update_adjacency_matrices(
                A_train_secondary,
                test_nodes_set
                | validation_nodes_set
                | not_chosen_in_train
                | not_chosen_in_val,
            )

            # We need updated indexes
            # Train
            remaining_nodes_train = all_nodes_set - (
                test_nodes_set
                | validation_nodes_set
                | not_chosen_in_train
                | not_chosen_in_val
            )
            index_mapping_train = {
                old_idx: new_idx
                for new_idx, old_idx in enumerate(sorted(remaining_nodes_train))
            }
            train_nodes_set_temp = chosen_set_train
            assert (
                len(remaining_nodes_train) == training_set.shape[1]
            ), "The number of nodes in the training set does not match the number of nodes in the all_nodes_set."
            missing_nodes = [
                node for node in train_nodes_set_temp if node not in index_mapping_train
            ]
            assert (
                not missing_nodes
            ), f"Some train nodes are missing from index_mapping_train: {missing_nodes}"
            train_nodes_set_updated = [
                index_mapping_train[node]
                for node in sorted(train_nodes_set_temp)
                if node in index_mapping_train
            ]
            # Validation
            remaining_nodes_val = all_nodes_set - (
                test_nodes_set | not_chosen_in_train | not_chosen_in_val
            )
            index_mapping_val = {
                old_idx: new_idx
                for new_idx, old_idx in enumerate(sorted(remaining_nodes_val))
            }
            validation_nodes_set_temp = chosen_set_val
            assert (
                len(remaining_nodes_val) == validation_set.shape[1]
            ), "The number of nodes in the validation set does not match the number of nodes in the all_nodes_set."
            missing_nodes = [
                node
                for node in validation_nodes_set_temp
                if node not in index_mapping_val
            ]
            assert (
                not missing_nodes
            ), f"Some train nodes are missing from index_mapping_train: {missing_nodes}"
            validation_nodes_set_updated = [
                index_mapping_val[node]
                for node in sorted(validation_nodes_set_temp)
                if node in index_mapping_val
            ]
            # Test
            remaining_nodes_test = all_nodes_set - (
                not_chosen_in_train | not_chosen_in_val
            )
            index_mapping_test = {
                old_idx: new_idx
                for new_idx, old_idx in enumerate(sorted(remaining_nodes_test))
            }
            test_nodes_set_temp = test_nodes_set.copy()
            assert (
                len(remaining_nodes_test) == test_set.shape[1]
            ), "The number of nodes in the test set does not match the number of nodes in the all_nodes_set."
            missing_nodes = [
                node for node in test_nodes_set_temp if node not in index_mapping_test
            ]
            assert (
                not missing_nodes
            ), f"Some train nodes are missing from index_mapping_train: {missing_nodes}"
            test_nodes_set_updated = [
                index_mapping_test[node]
                for node in sorted(test_nodes_set_temp)
                if node in index_mapping_test
            ]

    assert (
        max(validation_nodes_set_updated) <= validation_set.shape[1]
    ), "The new indexes are not correct."
    assert (
        max(test_nodes_set_updated) <= test_set.shape[1]
    ), "The new indexes are not correct."
    assert (
        max(train_nodes_set_updated) <= training_set.shape[1]
    ), "The new indexes are not correct."

    return (
        training_set,
        validation_set,
        test_set,
        A_test_primary,
        A_val_primary,
        A_train_primary,
        A_test_secondary,
        A_val_secondary,
        A_train_secondary,
        set(train_nodes_set_updated),
        set(validation_nodes_set_updated),
        set(test_nodes_set_updated),
        set(train_nodes_proof_of_concept),
        set(validation_nodes_proof_of_concept),
    )


def prepare_data_and_compute_matrices(
    test_nodes_set,
    percentage_of_segments_used_for_test,
    include_entire_graph_train_segments,
    include_entire_graph_test_segments,
    percentage_of_segments_used_for_training,
    device,
    feature_selection,
    only_extrapolation_and_no_forecasting,
    binary_or_distance_based_adjacency_matrix,
    percentage_of_the_training_data_used_for_validation,
    target_feature,
    cross_validation_group,
):
    """
    Prepares data, adjusts datasets, and computes adjacency matrices for training, validation, and testing.

    Args:
        test_nodes_set (set): Set of test nodes.
        feature_selection (str): Feature selection method.
        only_extrapolation_and_no_forecasting (bool): Whether to use only extrapolation.
        type_of_adjacency_matrix (str): Type of adjacency matrix.
        percentage_of_the_training_data_used_for_validation (float): Percentage of training data used for validation.
        target_feature (str): Target feature for prediction.
        percentage_of_segments_used_for_test (float): Percentage of segments used exclusively for testing.
        cross_validation_group (int): Cross-validation group index.
        include_entire_graph_train_segments (bool): Whether to include hidden training data.
        include_entire_graph_test_segments (bool): Whether to include the test sensor in training data.
        percentage_of_segments_used_for_training (float): Percentage of segments used for training.
        device (torch.device): Device to use for computation.

    Returns:
        tuple: Prepared datasets, adjacency matrices, and computed matrices.
    """

    (
        A_train_primary,
        A_val_primary,
        A_test_primary,
        A_train_secondary,
        A_val_secondary,
        A_test_secondary,
        training_set,
        validation_set,
        test_set,
        counting_station_names,
        test_nodes_set,
        all_nodes_set,
        train_nodes_set,
        validation_nodes_set,
    ) = get_or_load_data(
        test_nodes_set,
        feature_selection,
        only_extrapolation_and_no_forecasting,
        binary_or_distance_based_adjacency_matrix,
        percentage_of_the_training_data_used_for_validation,
        target_feature,
        percentage_of_segments_used_for_test,
        cross_validation_group,
    )

    # We sort the nodes set
    test_nodes_set = set(np.sort(list(test_nodes_set)))
    all_nodes_set = set(np.sort(list(all_nodes_set)))
    train_nodes_set = set(np.sort(list(train_nodes_set)))
    validation_nodes_set = set(np.sort(list(validation_nodes_set)))

    ###########################
    # Only use x% of the data for training
    # currently, we load 90% of the training data and we load 10% as the test data
    # This blocks enables one, to use less than 90% of the data for training
    # Either these elements are masked or completly removed from the training data
    ###########################

    (
        training_set,
        validation_set,
        test_set,
        A_test_primary,
        A_val_primary,
        A_train_primary,
        A_test_secondary,
        A_val_secondary,
        A_train_secondary,
        train_nodes_set_updated,
        validation_nodes_set_updated,
        test_nodes_set_updated,
        train_nodes_proof_of_concept,
        validation_nodes_proof_of_concept,
    ) = employ_the_right_percentage_of_training_data_and_maybe_exclude_test(
        include_entire_graph_train_segments,
        include_entire_graph_test_segments,
        test_set,
        validation_set,
        training_set,
        train_nodes_set,
        validation_nodes_set,
        test_nodes_set,
        all_nodes_set,
        A_test_primary,
        A_val_primary,
        A_train_primary,
        A_test_secondary,
        A_val_secondary,
        A_train_secondary,
        percentage_of_segments_used_for_training,
        percentage_of_segments_used_for_test,
    )

    #####
    # insert here the percentage os known and unknown
    # we do this here, as for Strava, we only have one iteration in this for loop (the others are for LOGO)
    #####

    assert (
        percentage_of_segments_used_for_test + percentage_of_segments_used_for_training
        <= 1
    ), "The sum of the percentage of segments used for training and the percentage of segments used exclusively for test is not 1. THis needs to be iimplemented"

    # We compute the random A_q and A_h matrices, for the test (as they won't change, already here)
    with torch.no_grad():
        A_q_test_primary, A_h_test_primary = obtain_A_q_and_A_h_and_edges(
            A_test_primary, device
        )
        A_q_val_primary, A_h_val_primary = obtain_A_q_and_A_h_and_edges(
            A_val_primary, device
        )
        A_q_test_secondary, A_h_test_secondary = obtain_A_q_and_A_h_and_edges(
            A_test_secondary, device
        )
        A_q_val_secondary, A_h_val_secondary = obtain_A_q_and_A_h_and_edges(
            A_val_secondary, device
        )

    return (
        training_set,
        validation_set,
        test_set,
        A_train_primary,
        A_train_secondary,
        A_q_test_primary,
        A_h_test_primary,
        A_q_val_primary,
        A_h_val_primary,
        A_q_test_secondary,
        A_h_test_secondary,
        A_q_val_secondary,
        A_h_val_secondary,
        validation_nodes_set_updated,
        test_nodes_set_updated,
        train_nodes_set,
        validation_nodes_set,
    )


def prepare_mean_and_std(training_set, device):
    # Compute mean and std for the first feature (assuming it's the one to be normalized)
    mean = training_set[..., 0].mean()
    std = training_set[..., 0].std()

    # Convert to PyTorch tensor for efficiency
    mean_torch = torch.tensor(mean, dtype=torch.float32)  # or match tensor dtype
    std_torch = torch.tensor(std, dtype=torch.float32)

    # Move to device
    mean_torch = mean_torch.to(device)
    std_torch = std_torch.to(device)
    return mean_torch, std_torch


def load_data(
    test_nodes_set,
    feature_selection,
    only_extrapolation_and_no_forecasting,
    percentage_of_the_training_data_used_for_validation,
    target_feature,
    counter_ids,
):
    """
    Load data

    Input: test_nodes_set: a set of integers, which represent the unknown counting stations
    feature_selection: a string, which represents the feature selection
    get_counter_number: a boolean, which indicates if the number of counting stations should be returned
    -------
    Returns (if get_counter_number = False):
    The test and the trianing data as well as other relevant data. All data is preprocessed and standardized (except for the target feature).
    A: adjacency matrix (in shape of (all_nodes_set_length, all_nodes_set_length))
    A_training: the observed adjacent matrix (not including the unknown set)
    training_X: processed data (in shape of (all_nodes_set_length, num_timesteps, num_features) for the training set
    test_X: processed data (in shape of (all_nodes_set_length, num_timesteps, num_features) for the test set
    test_nodes_set: the unknow set (a set of integers)
    counter_ids: the names of the counting stations (in the relevant order (alphabetical))
    -------
    Returns (if get_counter_number = True):
    The number of sensors available in the data (without any hidden or alike).
    """

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

    if os.path.exists(file_to_load_pkl):
        data = pd.read_pickle(file_to_load_pkl)
    else:
        create_data_for_GNNUI(feature_selection)
        data = pd.read_pickle(file_to_load_pkl)

    (
        training_set_X,
        validation_set_X,
        test_set_X,
        train_nodes_set,
        validation_nodes_set,
        all_nodes_set,
    ) = generate_training_and_test_set(
        data,
        feature_selection,
        counter_ids,
        test_nodes_set,
        only_extrapolation_and_no_forecasting,
        percentage_of_the_training_data_used_for_validation,
        standardize_columns=True,
        target_feature_input=target_feature,
    )

    training_set_X = np.transpose(training_set_X, (1, 0, 2))
    validation_set_X = np.transpose(validation_set_X, (1, 0, 2))
    test_set_X = np.transpose(test_set_X, (1, 0, 2))

    return (
        training_set_X,
        validation_set_X,
        test_set_X,
        counter_ids,
        test_nodes_set,
        all_nodes_set,
        train_nodes_set,
        validation_nodes_set,
    )
