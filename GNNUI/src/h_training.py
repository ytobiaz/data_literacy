import torch
import numpy as np
import gc
from h_utils import get_know_mask
from h_evaluation import (
    create_mask_missing_groundtruth,
    replace_nan_by_mean_or_by_zero,
    replace_hidden_by_mean_or_by_zero,
)
from h_assert_statements import assert_aspects_about_hiding_index_and_Mf_inputs
from h_adjacency_matrices import obtain_A_q_and_A_h_and_edges
from h_utils import (
    add_feature_indicating_hidden_observation,
    add_dimension_of_ones_to_account_for_the_added_feature_indicating_hidden_observations,
)
from h_evaluation import (
    turn_inputs_into_the_needed_format,
)
import random


def subsample_data_for_training(
    training_set,
    all_possible_t_random,
    batch_size,
    h,
    n_o_n_m,
    n_m,
    A_train_primary,
    A_train_secondary,
    device,
    choose_t_random_better,
    include_node_features_if_observation_are_masked_or_missing,
    do_IGNNK,
    include_node_features,
    mean_torch,
    std_torch,
):

    # Choose a random number at which the training data shall start.
    if choose_t_random_better:
        t_random = all_possible_t_random[:batch_size]
        all_possible_t_random = all_possible_t_random[batch_size:]
    else:
        t_random = np.random.randint(
            0, high=(training_set.shape[0] - h), size=batch_size, dtype="l"
        )

    assert (
        n_o_n_m <= training_set.shape[1]
    ), "The number chosen for n_o_n_m  is more than the number of nodes in the available data. Please choose a lower number."

    ###########################
    # Choose which nodes are visible/known to the model
    ###########################

    know_mask = get_know_mask(
        n_o_n_m,
        training_set,
    )

    ###########################
    # Compose the inputs data
    ###########################
    # Compose the feed batch (appending parts of the training set).
    feed_batch = (
        []
    )  # this is used to store the data for the batch, which will be turned into inputs
    for j in range(batch_size):
        feed_batch.append(
            training_set[t_random[j] : t_random[j] + h, :, :][:, list(know_mask), :]
        )
    inputs = torch.tensor(np.array(feed_batch), device=device)

    ###########################
    # Create a mask for missing entries (those that are truly nan in the data)
    ###########################

    mask_missing_groundtruth = create_mask_missing_groundtruth(inputs, device)

    ###########################
    # Replace the nans:
    # either we replace the -1 in Strava with 0
    # or we replace the nans with the mean of the non-zero values (for Strava and count)
    ###########################

    inputs = replace_nan_by_mean_or_by_zero(
        inputs,
        mask_missing_groundtruth,
    )

    ###########################
    # Obtain As and edge infos dynamic (so for the nodes considered in this iteration)
    ###########################
    # Obtain the dynamic adjacent matrix
    A_dynamic_primary = A_train_primary[list(know_mask), :][:, list(know_mask)]
    A_dynamic_secondary = A_train_secondary[list(know_mask), :][:, list(know_mask)]
    A_q_primary, A_h_primary = obtain_A_q_and_A_h_and_edges(A_dynamic_primary, device)
    A_q_secondary, A_h_secondary = obtain_A_q_and_A_h_and_edges(
        A_dynamic_secondary, device
    )

    ###########################
    # In each batch, hide different nodes and prep the data
    ###########################

    # Create a mask to hide nodes
    ###########################
    # Create a mask for the inputs we want to mask (n_m nodes).
    hiding_mask = torch.ones(inputs.shape, device=device)

    # Iterate of the batches.
    # Choose some nodes to be masked (n_m nodes) (so hidden from the model, on which we evaluate the performance).
    # So in every batch, we do hide different nodes.
    for j in range(batch_size):
        hiding_index = random.sample(range(0, n_o_n_m), n_m)
        hiding_mask[j, :, list(hiding_index), 0] = (
            0  # we set the target feature (0) (across all time steps h) to zero/the standardized zero if applicable
        )

    # Use the mask to hide the values in the input data (only on first dimension, as we do not want to effect the features).
    Mf_inputs = inputs.clone()
    Mf_inputs = replace_hidden_by_mean_or_by_zero(
        Mf_inputs,
        hiding_mask,
    )

    # assert statements
    assert_aspects_about_hiding_index_and_Mf_inputs(
        hiding_mask,
        hiding_index,
        h,
        j,
    )

    # define mask
    mask = mask_missing_groundtruth.to(device)

    # We get the outputs, the ground truth (this is includes missing values as 0 and also those which we have masked out.)
    outputs = inputs.to(device)

    del inputs
    gc.collect()
    torch.cuda.empty_cache()

    ###########################
    # Insert a feature, indicating whether an observation was masked or not
    ###########################
    # We want to add a feature, which indicates if a an observation is masked (hidden)
    # The shape of the tensors ic (inputs) and Mf_inputs is (batch_size, h, n_o_n_m, no_of_features)
    # Thus, we can simply add a feature, which is 1 if the observation is masked and 0 if it is not masked.
    # We add this feature to the end of the tensor (so it is the last feature).
    # We do this for both the inputs and the masked inputs.

    # add feature indicating hidden observation
    if include_node_features_if_observation_are_masked_or_missing:
        # if hidden
        Mf_inputs = add_feature_indicating_hidden_observation(Mf_inputs, hiding_mask)
        # inputs = add_feature_indicating_hidden_observation(inputs, hiding_mask)
        outputs = add_feature_indicating_hidden_observation(outputs, hiding_mask)

        # Clean up after feature addition
        torch.cuda.empty_cache()

        # add a dimension of all 1s to hiding_mask and mask (as this feature shall be treated like all others)
        hiding_mask = add_dimension_of_ones_to_account_for_the_added_feature_indicating_hidden_observations(
            hiding_mask
        )
        mask = add_dimension_of_ones_to_account_for_the_added_feature_indicating_hidden_observations(
            mask
        )

        # account for missing
        Mf_inputs = add_feature_indicating_hidden_observation(
            Mf_inputs, mask_missing_groundtruth
        )
        # inputs = account_for_missing_in_feature_indicating_hidden_observation(inputs, mask_missing_groundtruth)
        outputs = add_feature_indicating_hidden_observation(
            outputs, mask_missing_groundtruth
        )

        # Clean up after feature addition
        del mask_missing_groundtruth
        torch.cuda.empty_cache()

        # add a dimension of all 1s to hiding_mask and mask (as this feature shall be treated like all others)
        hiding_mask = add_dimension_of_ones_to_account_for_the_added_feature_indicating_hidden_observations(
            hiding_mask
        )
        mask = add_dimension_of_ones_to_account_for_the_added_feature_indicating_hidden_observations(
            mask
        )

    ###########################
    # Reshape the data as needed
    ###########################

    if do_IGNNK == False:
        Mf_inputs[..., 0] = (Mf_inputs[..., 0] - mean_torch) / std_torch
        # Mf_inputs[..., 0] = 2 * (Mf_inputs[..., 0] - mean_torch) / (std_torch - mean_torch) - 1

    Mf_inputs = turn_inputs_into_the_needed_format(include_node_features, Mf_inputs)

    ###########################
    # Return
    ###########################
    return (
        Mf_inputs,
        A_q_primary,
        A_h_primary,
        A_q_secondary,
        A_h_secondary,
        A_train_primary,
        A_train_secondary,
        mask,
        outputs,
        hiding_mask,
        training_set,
        all_possible_t_random,
        batch_size,
        h,
        n_o_n_m,
        n_m,
        device,
    )
