import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import random
import gc
import torch
from paths import MODEL_RESULTS

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
from logging_config import setup_logging

# Initialize logging
setup_logging()

from paths import MODEL_RESULTS
from h_utils import *
from h_loss_functions import *
from h_training import *
from h_early_stopping import *
from h_data_preprocessing import *
from h_assert_statements import *
from h_adjacency_matrices import *
from h_evaluation import *
from h_model import *


def train_GNNUI_model(
    n_o_n_m,
    h,
    z,
    K,
    n_m,
    num_epochs,
    learning_rate,
    batch_size,
    include_node_features,
    include_node_features_if_observation_are_masked_or_missing,
    feature_selection,
    only_extrapolation_and_no_forecasting,
    type_of_adjacency_matrix,
    include_entire_graph_test_segments,
    percentage_of_segments_used_for_test,
    percentage_of_segments_used_for_training,
    percentage_of_the_training_data_used_for_validation,
    include_entire_graph_train_segments,
    choose_t_random_better,
    target_feature,
    schedule_learning_rate=False,
    compute_train_error_on_hidden_counter_only=True,
    compute_test_error_on_hidden_counter_only=True,
    overfitting_drop_out_rate=0,
    weight_decay=1e-4,
    type_of_loss="ZINB",
    patience_for_early_stopping_epochs=10,
    cross_validation_group=0,
    device="cuda:1",
    do_IGNNK=False,
    only_compute_test_error=False,
):
    logging.info("Starting GNNUI model training...")
    """
    Train the GNNUI model with the specified parameters.

    Parameters:
    - n_o_n_m (int): Number of observed nodes.
    - h (int): Time horizon.
    - z (int): Hidden dimension size.
    - K (int): Number of diffusion steps.
    - n_m (int): Number of masked nodes.
    - num_epochs (int): Number of training epochs.
    - learning_rate (float): Learning rate for the optimizer.
    - batch_size (int): Batch size for training.
    - include_node_features (bool): Whether to include node features.
    - include_node_features_if_observation_are_masked_or_missing (bool): Whether to include a feature indicating masked observations.
    - feature_selection (str): Feature selection method.
    - only_extrapolation_and_no_forecasting (bool): Whether to use only extrapolation.
    - type_of_adjacency_matrix (str): Type of adjacency matrix.
    - include_entire_graph_test_segments (bool): Whether to include the test sensor in training data.
    - percentage_of_segments_used_for_test (float): Percentage of segments used exclusively for testing.
    - percentage_of_segments_used_for_training (float): Percentage of segments used for training.
    - percentage_of_the_training_data_used_for_validation (float): Percentage of training data used for validation.
    - include_entire_graph_train_segments (bool): Whether to include hidden training data.
    - choose_t_random_better (bool): Whether to choose random time steps better.
    - target_feature (str): Target feature for prediction.
    - schedule_learning_rate (bool, optional): Whether to schedule the learning rate. Default is False.
    - compute_train_error_on_hidden_counter_only (bool, optional): Whether to compute training error only on hidden counters. Default is True.
    - compute_test_error_on_hidden_counter_only (bool, optional): Whether to compute test error only on hidden counters. Default is True.
    - overfitting_drop_out_rate (float, optional): Dropout rate to prevent overfitting. Default is 0.
    - weight_decay (float, optional): Weight decay for the optimizer. Default is 1e-4.
    - type_of_loss (str, optional): Type of loss function. Default is "ZINB".
    - patience_for_early_stopping_epochs (int, optional): Patience for early stopping. Default is 10.
    - cross_validation_group (int, optional): Cross-validation group index. Default is 0.
    - device (str, optional): Device to use for training. Default is "not_set".
    - do_IGNNK (bool, optional): Whether to use IGNNK. Default is False.
    - only_compute_test_error (bool, optional): Whether to only compute test error. Default is False.

    Returns:
    - None
    """

    # Control the combination of parameters
    control_if_the_combination_of_parameters_is_possible(
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
    )

    ##############################################
    # Set some parameters
    ##############################################

    nll_loss = set_gaussian_loss()
    ACTIVATION_FUNCTION_LAST_LAYER = "linear"
    best_val_metric = float("inf")  # used to initiate the best_val_metric
    patience_counter = 0
    if "strava" in target_feature:
        number_of_total_edges = 4958
    elif "taxi" in target_feature:
        number_of_total_edges = 8156
    random.seed(42)
    torch.manual_seed(42)

    ##############################################
    # Generate cross validation groups, if needed; save test nodes for baseline and set device
    ##############################################

    counting_stations_index = generate_cross_validation_groups(
        number_of_total_edges,
        cross_validation_group,
        percentage_of_segments_used_for_test,
    )

    # Save the test node set, to be used for the baseline, save list to file.
    save_test_node_set_for_baseline(
        feature_selection, target_feature, counting_stations_index
    )

    device = set_device(device)

    for test_nodes_set in counting_stations_index:

        assert_statements_for_target_feature(test_nodes_set)

        ###########################
        # Data preperation
        ###########################

        (
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
        ) = prepare_data_and_compute_matrices(
            test_nodes_set,
            percentage_of_segments_used_for_test,
            include_entire_graph_train_segments,
            include_entire_graph_test_segments,
            percentage_of_segments_used_for_training,
            device,
            feature_selection,
            only_extrapolation_and_no_forecasting,
            type_of_adjacency_matrix,
            percentage_of_the_training_data_used_for_validation,
            target_feature,
            cross_validation_group,
        )

        ###########################
        # Initalize model
        ###########################

        GNN_model, optimizer, scheduler = initialize_model(
            type_of_loss,
            do_IGNNK,
            type_of_adjacency_matrix,
            include_node_features_if_observation_are_masked_or_missing,
            include_node_features,
            training_set,
            h,
            z,
            K,
            overfitting_drop_out_rate,
            ACTIVATION_FUNCTION_LAST_LAYER,
            learning_rate,
            weight_decay,
            schedule_learning_rate,
            device,
        )

        ###########################
        # Assert statements
        ###########################

        assert_the_data_is_correct_in_reference_to_extrapolation_and_masked_sensors(
            only_extrapolation_and_no_forecasting,
            include_entire_graph_test_segments,
            include_entire_graph_train_segments,
            training_set,
            validation_set,
            test_set,
            train_nodes_set,
            validation_nodes_set,
        )

        ###########################
        # All columns are already properly scaled
        # But when we use the counter values of the others, they still need to be scaled
        # So we need this here
        ###########################
        test_set = torch.from_numpy(test_set.astype("float16"))

        validation_set = torch.from_numpy(validation_set.astype("float16")).to(device)

        mean_torch, std_torch = prepare_mean_and_std(training_set, device)

        ###########################
        # Prep file to compute validation error
        ###########################

        (
            T_input_batches_val,
            mask_batches_val,
            truth_batches_val,
            mask_val,
            truth_val,
        ) = prep_test_error(
            set(validation_nodes_set_updated),
            validation_set,
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

        ########################################################################################
        # Training the model

        for epoch in range(num_epochs):

            logging.info(f"Epoch {epoch + 1}/{num_epochs}")
            # Using time_length as reference to record test_error (training_set.shape[0] is the time)
            # Create a list of all possible "start dates" for t_random
            all_possible_t_random = generate_random_list_of_all_possible_t_random(
                training_set.shape[0], h
            )

            for i in range(training_set.shape[0] // (h * batch_size)):
                GNN_model.train()

                # Get the data for the sample
                (
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
                ) = subsample_data_for_training(
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
                )

                ###########################
                # Train the model
                ###########################

                if type_of_loss == "ZINB":
                    n_train, p_train, pi_train = GNN_model(
                        Mf_inputs,
                        A_q_primary,
                        A_h_primary,
                        A_q_secondary,
                        A_h_secondary,
                    )
                    X_res = None
                    pred_mean = None
                    pred_var = None
                elif type_of_loss == "NB":
                    n_train, p_train = GNN_model(
                        Mf_inputs,
                        A_q_primary,
                        A_h_primary,
                        A_q_secondary,
                        A_h_secondary,
                    )
                    pi_train = None
                    X_res = None
                    pred_mean = None
                    pred_var = None
                elif type_of_loss in {
                    "MAE",
                    "RMSE",
                    "MSE",
                }:
                    X_res = GNN_model(
                        Mf_inputs,
                        A_q_primary,
                        A_h_primary,
                        A_q_secondary,
                        A_h_secondary,
                    )
                    n_train = None
                    p_train = None
                    pi_train = None
                    pred_mean = None
                    pred_var = None
                elif type_of_loss == "NLL":
                    pred_mean, pred_var = GNN_model(
                        Mf_inputs,
                        A_q_primary,
                        A_h_primary,
                        A_q_secondary,
                        A_h_secondary,
                    )
                del Mf_inputs

                ###########################
                # Compute the loss for back propagation
                ###########################

                loss = compute_the_loss(
                    compute_train_error_on_hidden_counter_only,
                    type_of_loss,
                    hiding_mask,
                    mask,
                    outputs,
                    device,
                    pred_mean,
                    pred_var,
                    n_train,
                    p_train,
                    pi_train,
                    nll_loss,
                    compute_weight_matrix_for_only_evaluating_on_left_out,
                    X_res,
                )

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                ###########################
                # Compute the loss to note it down in case the target was scaled.
                # WE then compute the loss on the "scaled back to original" data, to obtain a loss which is comparable to the test error.
                # At the end of the batches
                ###########################
                GNN_model.eval()

                ###########################
                # Compute validation error
                ###########################

                torch.cuda.empty_cache()
                (
                    MAE_val,
                    RMSE_val,
                    MSE_val,
                    prediction_last_one_val,
                    truth_last_one_val,
                    NLL_val,
                    mask,
                ) = test_error(
                    GNN_model,
                    A_q_val_primary,
                    A_h_val_primary,
                    A_q_val_secondary,
                    A_h_val_secondary,
                    device,
                    type_of_loss,
                    T_input_batches_val,
                    mask_batches_val,
                    truth_batches_val,
                    mask_val,
                    truth_val,
                )

                gc.collect()
                torch.cuda.empty_cache()

            ###########################
            # Evaluation for early stopping
            ###########################


            gc.collect()
            torch.cuda.empty_cache()
            metrics = {
                "MAE": MAE_val,
                "RMSE": RMSE_val,
                "MSE": MSE_val,
                "ZINB": NLL_val,
                "NB": NLL_val,
                "NLL": NLL_val,
            }

            val_metric = metrics.get(type_of_loss)

            if not val_metric > 0:
                raise ValueError("The val metric is nan")

            ##########################
            # update learnign rate
            ###########################

            if schedule_learning_rate:
                scheduler.step(val_metric)

            ##########################
            # Early stopping
            ###########################

            # Save model, with a name indicating all the parameters
            best_model_file_name = (
                "best_model_"
                + target_feature
                + str(include_node_features)[0]
                + str(include_node_features_if_observation_are_masked_or_missing)[0]
                + str(only_extrapolation_and_no_forecasting)[0]
                + str(type_of_adjacency_matrix)
                + str(include_entire_graph_test_segments)[0]
                + str(percentage_of_segments_used_for_test)
                + str(percentage_of_segments_used_for_training)
                + str(include_entire_graph_train_segments)[0]
                + str(choose_t_random_better)[0]
                + str(schedule_learning_rate)[0]
                + str(compute_train_error_on_hidden_counter_only)[0]
                + str(compute_test_error_on_hidden_counter_only)[0]
                + str(False)[0]
                + str(type_of_loss)
                + ".pth"
            )

            best_val_metric, patience_counter, stop_training = handle_early_stopping(
                val_metric,
                best_val_metric,
                patience_counter,
                patience_for_early_stopping_epochs,
                GNN_model,
                MODEL_RESULTS / best_model_file_name,
            )

            ##########################
            # Computation of test error
            ###########################

            if stop_training or epoch == num_epochs - 1 or only_compute_test_error:
                del (
                    T_input_batches_val,
                    mask_batches_val,
                    truth_batches_val,
                    mask_val,
                    truth_val,
                    training_set,
                    validation_set,
                )
                del (
                    A_q_val_primary,
                    A_h_val_primary,
                    A_q_val_secondary,
                    A_h_val_secondary,
                    A_train_primary,
                    A_train_secondary,
                )
                (
                    MAE_test_last,
                    RMSE_test_last,
                    NLL_test_last,
                    KL_Divergence,
                    true_zero_rate,
                ) = compute_final_test_error(
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
                )

                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        ###########################
        # Prepare what will be returned
        ###########################

    logging.info("Training completed.")
    return (
        MAE_test_last.item(),
        RMSE_test_last.item(),
        NLL_test_last,
        KL_Divergence,
        true_zero_rate,
    )
