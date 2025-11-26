from __future__ import division
import numpy as np
import torch
from torch_geometric.data import Dataset
import gc
import logging
import scipy.stats
from h_loss_functions import set_gaussian_loss, nb_nll_loss, nb_zeroinflated_nll_loss
from h_utils import (
    add_feature_indicating_hidden_observation,
    compute_loss,
    add_dimension_of_ones_to_account_for_the_added_feature_indicating_hidden_observations,
    free_memory,
)
from paths import MODEL_RESULTS

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def assert_Tinputs_correct_shape(T_inputs):
    assert len(T_inputs.shape) == 3, "The input data does not have the correct shape."


def turn_inputs_into_the_needed_format(include_node_features, inputs):
    # We need to reshape the input data, such that it fits the gnn_model.
    if not include_node_features:
        # If we only include the count values, we drop all the features.
        inputs = inputs[:, :, :, 0]  # We only keep the count values
    else:
        # If we include the node features, we need to reshape the data.
        # We add the features together in one dimension.
        inputs = inputs.permute(0, 1, 3, 2)
        inputs = inputs.reshape(
            inputs.shape[0], -1, inputs.shape[3]
        )  # here we combine h*f, such that [feature1,, Feautre f for day 1], [feature1,, Feautre f for day 2],...
    return inputs


def create_mask_missing_groundtruth(data, device, value_indicating_missing_data=-1):
    mask_missing_groundtruth = torch.ones(data.shape).to(device)

    # Define mask where entries are missing
    dimension_indices_with_focus_on_last = (slice(None),) * (data.dim() - 1) + (0,)
    # Apply the mask on the last dimension
    mask_missing_groundtruth[dimension_indices_with_focus_on_last][
        data[dimension_indices_with_focus_on_last] == value_indicating_missing_data
    ] = 0

    return mask_missing_groundtruth


def is_assert_prediction_ok(prediction) -> bool:
    return (
        len(prediction.shape) == 2
        and prediction.shape[0] > 0
        and prediction.shape[1] > 0
    )


def replace_nan_by_mean_or_by_zero(
    inputs,
    mask_missing_groundtruth,
):
    """
    Replace NaN or masked values in the input tensor based on the specified method.

    Parameters:
    - inputs (Tensor): The input tensor where replacements will be made.
    - use_mean_for_masking (bool): Whether to replace masked values with the mean.
    - target_feature (str): The target feature type (e.g., 'strava_total', 'count').
    Returns:
    - Tensor: The updated input tensor with masked values replaced.
    """
    inputs[..., 0] = torch.where(
        mask_missing_groundtruth[..., 0] == 0,
        torch.full_like(inputs[..., 0], 0),
        inputs[..., 0],
    )

    return inputs


def replace_hidden_by_mean_or_by_zero(data_input, hiding_mask):
    data_input[:, :, :, 0] = torch.where(
        hiding_mask[:, :, :, 0] == 0,
        torch.full_like(data_input[:, :, :, 0], 0),
        data_input[:, :, :, 0],
    )
    return data_input


def compute_weight_matrix_for_only_evaluating_on_left_out(
    hiding_index, mask_missing_groundtruth
):
    return (1 - hiding_index) * mask_missing_groundtruth


def prep_test_error(
    unknow_set,
    test_data,
    include_node_features,
    include_node_features_if_observation_are_masked_or_missing,
    device,
    compute_test_error_on_hidden_counter_only,
    only_extrapolation_and_no_forecasting,
    mean_torch,
    std_torch,
    h,
    do_IGNNK,
):
    # logging.info("Preparing test error...")
    #######################
    # Mask missing groundtruth
    #######################

    mask = create_mask_missing_groundtruth(test_data, device)

    #######################
    # Replace -1 by 0 (for Strava, indicating nan)
    #######################

    test_data = replace_nan_by_mean_or_by_zero(test_data, mask)

    #######################
    # Create the mask to hide the unknown values from the model
    #######################

    # The unknown set (the hidden one) is set to zero
    hiding_index = torch.ones(test_data.shape)
    # Maximum in unknow_Set
    hiding_index[:, list(unknow_set), 0] = 0
    # delete unknown set
    unknow_set = None
    # to device
    hiding_index = hiding_index.to(device)

    #######################
    # Put data in batched format
    #######################

    num_batches = test_data.shape[0] // h
    # Next we need to put the input data into the right shape (so batches each of size h)
    mask = mask[: num_batches * h].reshape(
        num_batches, h, test_data.shape[1], test_data.shape[2]
    )
    mask = mask[:, :, :, 0].permute(0, 2, 1)

    batched_inputs = test_data[: num_batches * h].reshape(
        num_batches, h, test_data.shape[1], test_data.shape[2]
    )
    batched_target = batched_inputs[:, :, :, 0]
    batched_hiding_index = hiding_index[: num_batches * h].reshape(
        num_batches, h, test_data.shape[1], test_data.shape[2]
    )
    truth = batched_target.permute(0, 2, 1)

    T_inputs = batched_inputs.clone()

    #######################
    # Add feature indicating if the value is masked
    ######################
    # Add the feature to the inputs
    if include_node_features_if_observation_are_masked_or_missing:

        T_inputs = add_feature_indicating_hidden_observation(
            T_inputs, batched_hiding_index
        )
        # Add a dimension of all 1s to hiding_mask and mask (as this feature shall be treated like all others)
        batched_hiding_index = add_dimension_of_ones_to_account_for_the_added_feature_indicating_hidden_observations(
            batched_hiding_index
        )

        # also for missing
        T_inputs = add_feature_indicating_hidden_observation(T_inputs, mask)
        batched_hiding_index = add_dimension_of_ones_to_account_for_the_added_feature_indicating_hidden_observations(
            batched_hiding_index
        )

    #######################
    # Prepare T Inputs
    #######################
    T_inputs = replace_hidden_by_mean_or_by_zero(
        T_inputs,
        batched_hiding_index,
    )

    if do_IGNNK == False:
        T_inputs[..., 0] = (T_inputs[..., 0] - mean_torch) / std_torch
        # T_inputs[..., 0] = 2 * (T_inputs[..., 0] - mean_torch) / (std_torch - mean_torch) - 1
    T_inputs = turn_inputs_into_the_needed_format(include_node_features, T_inputs)

    assert_Tinputs_correct_shape(T_inputs)
    #######################
    # compute weight matrix
    #######################

    if compute_test_error_on_hidden_counter_only:
        mask = compute_weight_matrix_for_only_evaluating_on_left_out(
            batched_hiding_index[:, :, :, 0].permute(0, 2, 1), mask
        )

    del batched_hiding_index

    #######################
    # Make prediction
    #######################

    # We need to do in parts as T_inputs is too large
    # Split it by 8
    # First dimesnion is
    if only_extrapolation_and_no_forecasting:
        num_splits = 4  # Number of chunks
    else:
        num_splits = 1

    T_input_batches = torch.chunk(T_inputs, num_splits, dim=0)
    mask_batches = torch.chunk(mask, num_splits, dim=0)
    truth_batches = torch.chunk(truth, num_splits, dim=0)

    # logging.info("Test error preparation completed.")
    return T_input_batches, mask_batches, truth_batches, mask, truth


def test_error(
    gnn_model,
    A_q_primary,
    A_h_primary,
    A_q_secondary,
    A_h_secondary,
    device,
    type_of_loss_ZINB_NB_classic,
    T_input_batches,
    mask_batches,
    truth_batches,
    mask,
    truth,
):
    # logging.info("Computing test error...")
    nll_loss = set_gaussian_loss()
    predictions = []
    ZINB_NB_ERRORS = []
    for T_batch, mask_batch, truth_batch in zip(
        T_input_batches, mask_batches, truth_batches
    ):
        # choose also right mask_batches and right truth_batches
        with torch.no_grad():
            if (
                type_of_loss_ZINB_NB_classic == "MAE"
                or type_of_loss_ZINB_NB_classic == "RMSE"
            ):
                prediction_batch = gnn_model(
                    T_batch, A_q_primary, A_h_primary, A_q_secondary, A_h_secondary
                ).permute(0, 2, 1)
                ZINB_test_error_if_applicable = 0
            elif type_of_loss_ZINB_NB_classic == "ZINB":
                n_test, p_test, pi_test = gnn_model(
                    T_batch, A_q_primary, A_h_primary, A_q_secondary, A_h_secondary
                )
                ZINB_test_error_if_applicable = nb_zeroinflated_nll_loss(
                    truth_batch, n_test, p_test, pi_test, mask_batch, device, "mean"
                ).item()
                prediction_batch = (1 - pi_test) * (
                    n_test / p_test - n_test
                )  # Expected value of ZINB distribution
            elif type_of_loss_ZINB_NB_classic == "NB":
                n_test, p_test = gnn_model(
                    T_batch, A_q_primary, A_h_primary, A_q_secondary, A_h_secondary
                )
                ZINB_test_error_if_applicable = nb_nll_loss(
                    truth_batch, n_test, p_test, mask_batch, "mean"
                ).item()
                prediction_batch = (
                    n_test / p_test - n_test
                )  # Expected value of NB distribution
            elif type_of_loss_ZINB_NB_classic == "NLL":
                prediction_batch, prediction_var = gnn_model(
                    T_batch, A_q_primary, A_h_primary, A_q_secondary, A_h_secondary
                )
                assert torch.all(
                    torch.logical_or(
                        mask_batch == 0,
                        mask_batch == 1,
                    )
                )
                # Create a mask for valid entries (where weight is 1)
                valid_mask = mask_batch == 1
                # Filter the y_batch and pred_mean by the valid mask (keep only valid entries)
                valid_truth_batch = truth_batch[valid_mask]
                valid_pred_batch = prediction_batch[valid_mask]
                valid_pred_var = prediction_var[
                    valid_mask
                ]  # Only if you're using pred_var
                # Compute the loss on the valid entries
                ZINB_test_error_if_applicable = nll_loss(
                    valid_pred_batch, valid_truth_batch, valid_pred_var
                )
                # convert from torch tensor to float
                ZINB_test_error_if_applicable = ZINB_test_error_if_applicable.item()

                # Los here is very similar with 14.513

        ZINB_NB_ERRORS.append(ZINB_test_error_if_applicable)
        predictions.append(prediction_batch.clone().detach())

    del T_input_batches

    prediction = torch.cat(predictions, dim=0)

    torch.cuda.empty_cache()
    gc.collect()
    #######################
    # Compute the error
    #######################

    with torch.no_grad():
        MAE = compute_loss(prediction, truth, mask, loss_type="MAE")
        RMSE = compute_loss(prediction, truth, mask, loss_type="RMSE")
        MSE = compute_loss(prediction, truth, mask, loss_type="MSE")
    # print('MAE:', MAE.item(), 'RMSE:', RMSE.item(), 'ZINB_NB_ERRORS:', ZINB_test_error_if_applicable)

    del predictions
    del ZINB_NB_ERRORS
    del T_batch, mask_batch, truth_batch
    # logging.info("Test error computation completed.")
    return (
        MAE,
        RMSE,
        MSE,
        prediction,
        truth,
        ZINB_test_error_if_applicable,
        mask,
    )


def compute_final_test_error(
    epoch,
    num_epochs,
    GNN_model,
    best_model_file_name,
    device,
    test_set,
    A_q_test_primary,
    A_h_test_primary,
    A_q_test_secondary,
    A_h_test_secondary,
    type_of_loss,
    test_nodes_set_updated,
    include_node_features,
    include_node_features_if_observation_are_masked_or_missing,
    compute_test_error_on_hidden_counter_only,
    only_extrapolation_and_no_forecasting,
    mean_torch,
    std_torch,
    h,
    do_IGNNK,
):

    free_memory()

    if epoch == num_epochs - 1:
        print(
            "Reached maximum number of epochs without reaching the minimum validation error."
        )

    GNN_model.load_state_dict(torch.load(MODEL_RESULTS / best_model_file_name))

    test_set = test_set.to(device)
    A_q_test_primary = A_q_test_primary.to(device)
    A_h_test_primary = A_h_test_primary.to(device)
    A_q_test_secondary = A_q_test_secondary.to(device)
    A_h_test_secondary = A_h_test_secondary.to(device)
    std_torch = std_torch.to(device)
    mean_torch = mean_torch.to(device)
    GNN_model = GNN_model.to(device)

    (
        T_input_batches_test,
        mask_batches_test,
        truth_batches_test,
        mask_test,
        truth_test,
    ) = prep_test_error(
        set(test_nodes_set_updated),
        test_set,  # validation_set,
        include_node_features,
        include_node_features_if_observation_are_masked_or_missing,
        device,
        compute_test_error_on_hidden_counter_only,
        only_extrapolation_and_no_forecasting,
        mean_torch,
        std_torch,
        h,
        do_IGNNK,
    )

    (
        MAE_test_last,
        RMSE_test_last,
        MSE_test_last,
        prediction_last_one_test_last,
        truth_last_one_test_last,
        NLL_test_last,
        mask,
    ) = test_error(
        GNN_model,
        A_q_test_primary,
        A_h_test_primary,
        A_q_test_secondary,
        A_h_test_secondary,
        device,
        type_of_loss,
        T_input_batches_test,
        mask_batches_test,
        truth_batches_test,
        mask_test,
        truth_test,
    )

    print(
        "the test errors are: MAE, RMSE, NLL_test_last:",
        MAE_test_last,
        RMSE_test_last,
        NLL_test_last,
    )

    ###########################
    # Compute additional scores
    ###########################

    KL_Divergence = compute_KL_Divergence(
        prediction_last_one_test_last, truth_last_one_test_last, mask
    )
    true_zero_rate = compute_true_zero_rate(
        prediction_last_one_test_last, truth_last_one_test_last, mask
    )

    return (
        MAE_test_last,
        RMSE_test_last,
        NLL_test_last,
        KL_Divergence,
        true_zero_rate,
    )


def compute_KL_Divergence(prediction, truth, weights, num_bins=100, epsilon=1e-10):
    logging.debug("Computing KL Divergence...")
    """
    Computes the Kullback-Leibler (KL) Divergence between two probability distributions.

    KL Divergence measures how one probability distribution differs from another.

    Parameters:
    - prediction (np.array or torch.Tensor): Predicted values.
    - truth (np.array or torch.Tensor): Ground truth values.
    - weights (np.array or torch.Tensor): Masking weights (only 1s are considered).
    - num_bins (int, optional): Number of bins for histogram estimation. Default is 100.
    - epsilon (float, optional): Small value to avoid log(0) errors. Default is 1e-10.

    Returns:
    - float: KL divergence value.
    """

    # Convert tensors to NumPy arrays if needed
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(truth, torch.Tensor):
        truth = truth.cpu().numpy()
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()

    # Filter only the observations where weights == 1
    mask = weights == 1
    prediction_filtered = prediction[mask]
    truth_filtered = truth[mask]

    # Define bin edges based on min/max of filtered data
    min_val = min(prediction_filtered.min(), truth_filtered.min())
    max_val = max(prediction_filtered.max(), truth_filtered.max())
    bins = np.linspace(min_val, max_val, num_bins + 1)

    # Compute histograms (density=True normalizes bin widths, but we normalize manually)
    P_hist, _ = np.histogram(truth_filtered, bins=bins, density=True)
    Q_hist, _ = np.histogram(prediction_filtered, bins=bins, density=True)

    # Normalize to get probability distributions
    P_prob = P_hist / P_hist.sum()
    Q_prob = Q_hist / Q_hist.sum()

    # Add small epsilon to avoid log(0) issues
    P_prob = np.clip(P_prob, epsilon, 1)
    Q_prob = np.clip(Q_prob, epsilon, 1)

    # Compute KL divergence
    kl_div = scipy.stats.entropy(P_prob, Q_prob)

    logging.debug(f"KL Divergence computed: {kl_div}")
    return kl_div


def compute_true_zero_rate(prediction, truth, weights, tau=0.99):
    """
    Computes the True Zero Rate, which measures how often a model correctly identifies true zero values.

    Parameters:
    - prediction (torch.Tensor): Model predictions.
    - truth (torch.Tensor): Ground truth values.
    - weights (torch.Tensor): Weights indicating valid samples (typically binary: 1 for valid, 0 for ignored).
    - tau (float, optional): Threshold to consider a predicted value as zero. Default is 0.99.

    Returns:
    - float: Ratio of correctly identified true zeros to total true zeros.
    """
    # Identify true zero values in the ground truth
    true_zeros = (torch.abs(truth) == 0).float() * weights

    # Count correctly predicted zeros (within tau threshold)
    correct_zeros = ((torch.abs(prediction) < tau) * true_zeros).sum()

    # Total number of true zeros
    total_true_zeros = true_zeros.sum()

    return correct_zeros / (total_true_zeros)
