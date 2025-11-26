import torch
import numpy as np
import pandas as pd

from paths import MODEL_RESULTS
from h_model import initialize_model
from h_data_preprocessing import prepare_data_and_compute_matrices, prepare_mean_and_std
from h_evaluation import prep_test_error, test_error
from h_adjacency_matrices import load_adjacency_matrix_and_counter_ids
from h_utils import generate_cross_validation_groups, set_device


def export_strava_segment_densities(
    output_csv_path: str = "strava_segment_densities.csv",
    device_str: str = "cpu",  # change to "cuda:0" if you like
):
    """
    Load the pretrained Strava GNNUI model and export average predicted traffic
    volume (over the test period) for each street segment.
    """
    # ------------------------------------------------------------------
    # 1. Config – matching the training run for the Strava model
    # ------------------------------------------------------------------
    n_o_n_m = 2500
    h = 2
    z = 100
    K = 1

    feature_selection = "full_data_all_features_for_strava_as_target"
    only_extrapolation_and_no_forecasting = True
    type_of_adjacency_matrix = "binary"
    include_entire_graph_test_segments = True
    percentage_of_segments_used_for_test = 0.1
    percentage_of_segments_used_for_training = 0.9
    percentage_of_the_training_data_used_for_validation = 0.1
    include_entire_graph_train_segments = True
    target_feature = "strava_total_trip_count"

    include_node_features = True
    include_node_features_if_observation_are_masked_or_missing = True
    compute_test_error_on_hidden_counter_only = True

    schedule_learning_rate = True
    type_of_loss = "ZINB"
    overfitting_drop_out_rate = 0.0
    weight_decay = 0.0
    learning_rate = 0.0001
    do_IGNNK = False
    cross_validation_group = 0  # same as in train.ipynb
    choose_t_random_better = True  # not used for test

    # Matches the hardcoded values in train_GNNUI_model
    if "strava" in target_feature:
        number_of_total_edges = 4958
    else:
        raise ValueError("This helper is configured only for the Strava model.")

    device = set_device(device_str)

    # ------------------------------------------------------------------
    # 2. Recreate the same test node set as in training
    # ------------------------------------------------------------------
    counting_stations_index = generate_cross_validation_groups(
        number_of_total_edges,
        cross_validation_group,
        percentage_of_segments_used_for_test,
    )
    # For cross_validation_group == 0, use the first entry
    test_nodes_set = counting_stations_index[cross_validation_group]

    # ------------------------------------------------------------------
    # 3. Load data + adjacency matrices
    # ------------------------------------------------------------------
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

    # Convert test set to torch
    test_set = torch.from_numpy(test_set.astype("float32")).to(device)

    # Mean/std for internal scaling
    mean_torch, std_torch = prepare_mean_and_std(training_set, device)

    # ------------------------------------------------------------------
    # 4. Initialize model and load pretrained weights
    # ------------------------------------------------------------------
    ACTIVATION_FUNCTION_LAST_LAYER = "linear"

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

    best_model_file_name = (
        "best_model_strava_total_trip_countTTTbinaryT0.10.9TTTTTFZINB.pth"
    )
    state_dict = torch.load(MODEL_RESULTS / best_model_file_name, map_location=device)
    GNN_model.load_state_dict(state_dict)
    GNN_model.to(device).eval()

    # Move adjacency matrices to device
    A_q_test_primary = A_q_test_primary.to(device)
    A_h_test_primary = A_h_test_primary.to(device)
    A_q_test_secondary = A_q_test_secondary.to(device)
    A_h_test_secondary = A_h_test_secondary.to(device)

    # ------------------------------------------------------------------
    # 5. Build test batches and run inference
    # ------------------------------------------------------------------
    (
        T_input_batches_test,
        mask_batches_test,
        truth_batches_test,
        mask_test,
        truth_test,
    ) = prep_test_error(
        set(test_nodes_set_updated),
        test_set,
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
        MAE,
        RMSE,
        MSE,
        prediction,
        truth,
        ZINB_test_error_if_applicable,
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

    # ------------------------------------------------------------------
    # 6. Aggregate predictions per street segment
    # ------------------------------------------------------------------
    # prediction / truth / mask shapes: (T, N, 1) or (T, N)
    pred_np = prediction.detach().cpu().numpy()
    truth_np = truth.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    # Squeeze last dim if necessary
    if pred_np.ndim == 3:
        pred_np = np.squeeze(pred_np, axis=-1)
    if truth_np.ndim == 3:
        truth_np = np.squeeze(truth_np, axis=-1)

    # Only keep valid positions (mask == 1); others become NaN
    valid_mask = mask_np == 1
    pred_valid = np.where(valid_mask, pred_np, np.nan)
    truth_valid = np.where(valid_mask, truth_np, np.nan)

    # Mean over time for each node (ignoring NaNs)
    mean_pred_per_node = np.nanmean(pred_valid, axis=0)
    mean_truth_per_node = np.nanmean(truth_valid, axis=0)

    # ------------------------------------------------------------------
    # 7. Map node indices to street segment IDs (counter names)
    # ------------------------------------------------------------------
    (
        _A_train_primary,
        _A_val_primary,
        _A_test_primary,
        _A_train_secondary,
        _A_val_secondary,
        _A_test_secondary,
        counting_station_names,
    ) = load_adjacency_matrix_and_counter_ids(
        feature_selection, type_of_adjacency_matrix, target_feature
    )

    if len(counting_station_names) != mean_pred_per_node.shape[0]:
        raise RuntimeError(
            f"Number of nodes in predictions ({mean_pred_per_node.shape[0]}) "
            f"does not match number of station IDs ({len(counting_station_names)})."
        )

    df = pd.DataFrame(
        {
            "segment_id": counting_station_names,
            "mean_predicted_volume_test_period": mean_pred_per_node,
            "mean_observed_volume_test_period": mean_truth_per_node,
        }
    )

    df.to_csv(output_csv_path, index=False)
    print(f"Saved segment-level densities to {output_csv_path}")


if __name__ == "__main__":
    export_strava_segment_densities()
