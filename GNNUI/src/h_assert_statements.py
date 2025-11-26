import torch
from itertools import chain
import logging
import numpy as np

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def assert_statements_on_counting_station_index_two(
    counting_stations_index,
    number_of_total_edges,
    percentage_of_segments_used_for_test,
):
    """
    Assert that the counting station index satisfies the expected conditions.

    Parameters:
    - counting_stations_index (list): List of counting station indices.
    - number_of_total_edges (int): Total number of edges.
    - percentage_of_segments_used_for_test (float): Percentage of segments used for testing.

    Raises:
    - AssertionError: If any condition is violated.
    """
    logging.info("Asserting counting station index...")

    # Check if the size of the first group matches the expected test segment size
    expected_test_size = int(
        number_of_total_edges * percentage_of_segments_used_for_test
    )
    actual_test_size = len(counting_stations_index[0])
    assert (
        abs(actual_test_size - expected_test_size) <= 1
    ), f"Expected test size ({expected_test_size}) does not match actual size ({actual_test_size})."

    # Ensure there is only one group in the index
    assert (
        len(counting_stations_index) == 1
    ), f"Expected exactly one group in counting_stations_index, but found {len(counting_stations_index)}."

    # Ensure no duplicate entries across all groups
    flattened_indices = list(chain.from_iterable(counting_stations_index))
    assert len(flattened_indices) == len(
        set(flattened_indices)
    ), "Duplicate entries found in counting_stations_index."

    logging.info("Counting station index assertion passed.")


def assert_weight_tensor_fulfills_requirements(weight_tensor_for_loss):
    """
    Assert that the weight tensor satisfies the required conditions.

    Parameters:
    - weight_tensor_for_loss (torch.Tensor): A 4D tensor representing weights for loss computation.

    Raises:
    - AssertionError: If any condition is violated.
    """
    logging.info("Asserting weight tensor requirements...")

    # Ensure all elements in the last dimensions (except the first) are 1
    assert torch.all(
        weight_tensor_for_loss[:, :, :, 1:] == 1
    ), "All elements in weight_tensor_for_loss[:, :, :, 1:] must be 1."

    # Ensure no missing values in the first dimension
    assert not torch.any(
        weight_tensor_for_loss[:, :, :, 0] == 0
    ), "No elements in weight_tensor_for_loss[:, :, :, 0] should be 0."

    logging.info("Weight tensor requirements assertion passed.")


def assert_aspects_about_hiding_index_and_Mf_inputs(
    hiding_mask,
    hiding_index,
    h,
    j,
):
    """
    Assert that the hiding mask and index satisfy the expected conditions.

    Parameters:
    - hiding_mask (torch.Tensor): A 4D tensor representing the hiding mask.
    - hiding_index (list): List of indices representing hidden elements.
    - h (int): Time horizon or sequence length.
    - j (int): Index of the current batch or sample.

    Raises:
    - AssertionError: If any condition is violated.
    """
    # logging.info("Asserting aspects about hiding index and Mf inputs...")

    # Check if the number of non-1 elements in the hiding mask matches the expected count
    non_one_elements = len(hiding_mask[j, :, :, :][hiding_mask[j, :, :, :] != 1])
    expected_hidden_elements = len(hiding_index) * h
    assert (
        non_one_elements == expected_hidden_elements
    ), f"The number of hiding elements ({non_one_elements}) does not match the expected count ({expected_hidden_elements})."

    # Ensure all elements in the last dimensions (except the first) are 1
    assert torch.all(
        hiding_mask[:, :, :, 1:] == 1
    ), "The hiding mask contains non-1 entries in dimensions other than the first."

    # logging.info("Aspects about hiding index and Mf inputs assertion passed.")


def assert_the_data_is_correct_in_reference_to_extrapolation_and_masked_sensors(
    only_extrapolation_and_no_forecasting,
    include_entire_graph_test_segments,
    include_entire_graph_train_segments,
    training_set,
    validation_set,
    test_set,
    train_nodes_set,
    validation_nodes_set,
):
    """
    Assert that the data is consistent with the configuration for extrapolation and masked sensors.

    Parameters:
    - only_extrapolation_and_no_forecasting (bool): Whether only extrapolation is used.
    - include_entire_graph_test_segments (bool): Whether the test sensor is included in the training data as a masked element.
    - include_entire_graph_train_segments (bool): Whether hidden training data is included as a masked element.
    - training_set (torch.Tensor): The training dataset.
    - validation_set (torch.Tensor): The validation dataset.
    - test_set (torch.Tensor): The test dataset.
    - train_nodes_set (list): List of training node indices.
    - validation_nodes_set (list): List of validation node indices.

    Raises:
    - AssertionError: If any condition is violated.
    """
    logging.info(
        "Asserting data correctness in reference to extrapolation and masked sensors..."
    )

    if only_extrapolation_and_no_forecasting:
        if not include_entire_graph_test_segments:
            if (
                include_entire_graph_test_segments
                and include_entire_graph_train_segments
            ):
                assert np.all(
                    training_set[:, :, :]
                    == test_set[:, list(train_nodes_set), :]
                    == validation_set[:, list(train_nodes_set), :]
                ), "The training set and the test set are not the same, but they should be at the mentioned points."
                assert np.all(
                    validation_set[:, :, :]
                    == test_set[:, list(validation_nodes_set), :]
                ), "The validation set and the test set are not the same, but they should be at the mentioned points."
        else:
            assert (
                training_set.shape == test_set.shape == validation_set.shape
            ), "The training, validation, and test sets do not have the same shape."
            assert np.all(
                training_set[:, list(train_nodes_set), :]
                == test_set[:, list(train_nodes_set), :]
            ), "The training set and the test set are not the same, but they should be at the mentioned points."
            assert np.all(
                validation_set[:, list(train_nodes_set), :]
                == training_set[:, list(train_nodes_set), :]
            ), "The validation set and the training set are not the same, but they should be at the mentioned points."
            assert np.all(
                validation_set[:, list(validation_nodes_set), :]
                == test_set[:, list(validation_nodes_set), :]
            ), "The validation set and the test set are not the same, but they should be at the mentioned points."
            assert np.all(
                training_set[:, list(train_nodes_set), 0]
                == test_set[:, list(train_nodes_set), 0]
            ), "The training set and the test set are not the same for the first feature, but they should be."
            assert np.all(
                validation_set[:, list(train_nodes_set), 0]
                == test_set[:, list(train_nodes_set), 0]
            ), "The validation set and the test set are not the same for the first feature, but they should be."
            assert np.all(
                validation_set[:, list(validation_nodes_set), 0]
                == test_set[:, list(validation_nodes_set), 0]
            ), "The validation set and the test set are not the same for the first feature, but they should be."

    logging.info(
        "Data correctness in reference to extrapolation and masked sensors assertion passed."
    )


def assert_statements_for_target_feature(counter_left_out):
    """
    Assert that the target feature satisfies the required conditions.

    Parameters:
    - counter_left_out (list): A list of counters left out for validation or testing.

    Raises:
    - AssertionError: If any condition is violated.
    """
    logging.info("Asserting target feature statements...")

    # Ensure counter_left_out is a list
    assert isinstance(
        counter_left_out, list
    ), f"Expected counter_left_out to be a list, but got {type(counter_left_out).__name__}."

    # Ensure the list has more than one element
    assert (
        len(counter_left_out) > 1
    ), f"Expected counter_left_out to have more than one element, but got {len(counter_left_out)}."

    logging.info("Target feature statements assertion passed.")


def control_if_the_combination_of_parameters_is_possible(
    percentage_of_segments_used_for_training,
    percentage_of_segments_used_for_test,
    target_feature,
    feature_selection,
    n_m,
    n_o_n_m,
    compute_train_error_on_hidden_counter_only,
    compute_test_error_on_hidden_counter_only,
    include_node_features_if_observation_are_masked_or_missing,
    include_node_features,
    cross_validation_group,
):
    """
    Assert that the combination of parameters is valid.

    Parameters:
    - percentage_of_segments_used_for_training (float): Percentage of segments used for training.
    - percentage_of_segments_used_for_test (float): Percentage of segments used exclusively for testing.
    - target_feature (str): Target feature for prediction.
    - feature_selection (str): Feature selection method.
    - n_m (int): Number of masked nodes.
    - n_o_n_m (int): Number of observed nodes.
    - compute_train_error_on_hidden_counter_only (bool): Whether to compute training error only on hidden counters.
    - compute_test_error_on_hidden_counter_only (bool): Whether to compute test error only on hidden counters.
    - include_node_features_if_observation_are_masked_or_missing (bool): Whether to include a feature indicating masked observations.
    - include_node_features (bool): Whether to include node features.
    - cross_validation_group (int): Number of cross-validation groups.

    Raises:
    - AssertionError: If any condition is violated.
    """
    logging.info("Controlling if the combination of parameters is possible...")

    # Ensure the sum of training and testing percentages does not exceed 1
    assert (
        percentage_of_segments_used_for_training + percentage_of_segments_used_for_test
    ) <= 1, (
        "The sum of percentage_of_segments_used_for_training and "
        "percentage_of_segments_used_for_test cannot exceed 1."
    )

    # Ensure feature selection matches the target feature
    if "strava_" in target_feature:
        assert (
            "strava" in feature_selection
        ), "Feature selection must include 'strava' for the chosen target feature."

    # Ensure the number of masked nodes does not exceed the number of observed nodes
    assert (
        n_m <= n_o_n_m
    ), f"The number of masked nodes (n_m={n_m}) cannot exceed the number of observed nodes (n_o_n_m={n_o_n_m})."

    # Log warnings for specific configurations
    if not compute_test_error_on_hidden_counter_only:
        logging.warning(
            "Test errors are computed on all counters, which differs from the usual expectation."
        )
    if target_feature == "count":
        logging.warning(
            "The target feature is 'count'. The percentages of segments used for training and testing are not relevant."
        )

    # Ensure node features are included if masked observation features are used
    if include_node_features_if_observation_are_masked_or_missing:
        assert (
            include_node_features
        ), "Node features must be included if a feature indicating masked observations is used."

    # Ensure cross-validation configuration is valid
    multiplication = cross_validation_group * percentage_of_segments_used_for_test
    assert multiplication <= 1, (
        f"Invalid configuration: cross_validation_group ({cross_validation_group}) * "
        f"percentage_of_segments_used_for_test ({percentage_of_segments_used_for_test}) = {multiplication}, "
        "which exceeds 1. This means some test data is reused."
    )
    if multiplication != 1:
        logging.warning(
            f"Using {percentage_of_segments_used_for_test * 100:.2f}% of the total data as the test set, "
            f"but there are {cross_validation_group} cross-validation groups. Some data might not be used for testing."
        )

    logging.info("Combination of parameters control passed.")
