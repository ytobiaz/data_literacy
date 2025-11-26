import random
import logging
import numpy as np
from h_assert_statements import assert_statements_on_counting_station_index_two
import torch
import pickle as pkl
import gc
import warnings
from paths import GNNUI_BASELINES
from h_constants import DEFAULT_EPSILON


def free_memory():
    """
    Frees up GPU and CPU memory by clearing caches, deleting tensors, and invoking garbage collection.
    """
    # Clear GPU memory
    torch.cuda.empty_cache()
# Delete all tensors in the current scope
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                del obj
        except Exception as e:
            pass
# Run garbage collection
    gc.collect()


def set_device(device: str = "auto") -> torch.device:
    """
    Set the device for PyTorch.

    Args:
        device (str): Device to set. Options are "auto", "cpu", "cuda:0", "cuda:1", etc.
                      Default is "auto", which selects the first available GPU or falls back to CPU.

    Returns:
        torch.device: The selected device.
    """
    if device == "auto":
        # Automatically select the first available GPU, or fallback to CPU
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    elif device == "cpu":
        return torch.device("cpu")
    elif device.startswith("cuda"):
        # Check if the specified GPU is available
        gpu_index = int(device.split(":")[1])
        if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
            return torch.device(device)
        else:
            warnings.warn(f"Warning: {device} is not available. Falling back to CPU.")
            return torch.device("cpu")
    else:
        raise ValueError(f"Invalid device option: {device}")


def save_test_node_set_for_baseline(
    feature_selection, target_feature, counting_stations_index
):
    if "full_data" in feature_selection:
        with open(
            GNNUI_BASELINES / f"test_nodes_set_{target_feature}.pkl", "wb"
        ) as file:
            pkl.dump(counting_stations_index, file)


def generate_cross_validation_groups(
    number_of_total_edges: int,
    cross_validation_group: int,
    percentage_of_segments_used_for_test: float,
) -> list:
    """
    Generate cross-validation groups by randomizing edges and selecting a subset for testing.

    Args:
        number_of_total_edges (int): Total number of edges.
        cross_validation_group (int): The index of the cross-validation group.
        percentage_of_segments_used_for_test (float): Percentage of edges to use for testing.

    Returns:
        list: A list of indices representing the test set for the specified cross-validation group.
    """
    # Randomize the order in range(0, number_of_total_edges)
    list_of_all_edges = list(range(0, number_of_total_edges))
    # randomize order
    random.shuffle(list_of_all_edges)
    # for i in NUMBER_OF_CROSS_VALIDATION_GROUPS , choose the first 10% of the list as the test set
    counting_stations_index = []
    for i in [cross_validation_group]:
        counting_stations_index_new = list_of_all_edges[
            int(i * number_of_total_edges * percentage_of_segments_used_for_test) : int(
                (i + 1) * number_of_total_edges * percentage_of_segments_used_for_test
            )
        ]
        counting_stations_index.append(counting_stations_index_new)

    assert_statements_on_counting_station_index_two(
        counting_stations_index,
        number_of_total_edges,
        percentage_of_segments_used_for_test,
    )
    return counting_stations_index


def get_know_mask(n_o_n_m, training_set):
    # logging.info(f"Generating known mask for {n_o_n_m} nodes...")
    length = len(training_set[1])
    know_mask = set(random.sample(range(length), n_o_n_m))
    # logging.info("Known mask generated.")
    return know_mask


def generate_random_list_of_all_possible_t_random(max_value, h):
    logging.info(
        f"Generating random list of possible t_random values with max_value={max_value} and h={h}..."
    )
    possible_indices = np.arange(0, max_value - h, h)
    np.random.shuffle(possible_indices)
    logging.info("Random list of t_random values generated.")
    return possible_indices


# Add the feature to the inputs indicating hidden observation
def add_feature_indicating_hidden_observation(
    data_input, mask_according_to_which_hidden
):
    # Check the shape of mask and reshape accordingly
    if mask_according_to_which_hidden.ndim == 3:  # If it's like mask_original
        # Reshape mask from (batch_size, nodes, time) to (batch_size, time, nodes, 1)
        mask_according_to_which_hidden = mask_according_to_which_hidden.permute(
            0, 2, 1
        ).unsqueeze(3)
    else:
        # Already in the correct format, just add a dimension for concatenation
        mask_according_to_which_hidden = mask_according_to_which_hidden[
            :, :, :, 0
        ].unsqueeze(3)

    return torch.cat((data_input, mask_according_to_which_hidden), dim=3)


def compute_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    weights: torch.Tensor,
    loss_type: str,
    epsilon: float = DEFAULT_EPSILON,
) -> torch.Tensor:
    """
    Computes the weighted loss based on the specified loss type.

    Parameters:
    - prediction: The model's predicted values.
    - truth: The ground truth values.
    - weights: Weight tensor for missing values and desired weighting.
    - loss_type: Type of loss to compute. Options are "MAE", "MAPE", "RMSE", "SMAPE", "MSE".
    - epsilon: Small constant to avoid division by zero in MAPE.

    Returns:
    - Computed loss based on the specified type.
    """
    # logging.info(f"Computing {loss_type} loss...")
    if loss_type == "MAE":
        # Weighted Mean Absolute Error
        loss = torch.sum(torch.abs(prediction - truth) * weights) / torch.sum(weights)

    elif loss_type == "MAPE":
        # Weighted Mean Absolute Percentage Error
        # Adding epsilon to the denominator to avoid division by zero
        # We have so many zero observations, that MAPE just makes no sense. Thus, we mask those when computing the loss.
        mask = truth != 0
        masked_truth = truth[mask]
        masked_prediction = prediction[mask]
        masked_weights = weights[mask]

        # Compute MAPE only on valid observations
        loss = (
            torch.sum(
                torch.abs((masked_prediction - masked_truth) / (masked_truth + epsilon))
                * masked_weights
            )
            / torch.sum(masked_weights)
        ) * 100

    elif loss_type == "RMSE":
        # Weighted Root Mean Squared Error
        loss = torch.sqrt(
            torch.sum((prediction - truth) ** 2 * weights) / torch.sum(weights)
        )

    elif loss_type == "SMAPE":
        # Weighted Symmetric Mean Absolute Percentage Error
        loss = (
            torch.sum(
                (
                    (torch.abs(prediction - truth))
                    / ((torch.abs(prediction) + torch.abs(truth)) / 2 + epsilon)
                )
                * weights
            )
            / torch.sum(weights)
            * 100
        )

    elif loss_type == "MSE":
        # Weighted Mean Squared Error
        loss = torch.sum((prediction - truth) ** 2 * weights) / torch.sum(weights)

    else:
        raise ValueError(
            "Invalid loss_type. Choose from 'MAE', 'MAPE', 'RMSE', 'SMAPE', 'MSE'."
        )

    # logging.info(f"{loss_type} loss computed.")
    return loss


def add_dimension_of_ones_to_account_for_the_added_feature_indicating_hidden_observations(
    data_input,
):
    return torch.cat(
        (data_input, torch.ones_like(data_input[:, :, :, 0]).unsqueeze(3)), dim=3
    )
