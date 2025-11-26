import torch


def handle_early_stopping(
    val_metric, best_val_metric, patience_counter, patience_limit, model, save_path
):
    """
    Handles early stopping logic, including saving the best model.

    Args:
        val_metric (float): Current validation metric.
        best_val_metric (float): Best validation metric so far.
        patience_counter (int): Current patience counter.
        patience_limit (int): Maximum patience before stopping.
        model (torch.nn.Module): The model being trained.
        save_path (str): Path to save the best model.

    Returns:
        tuple: Updated best_val_metric, patience_counter, and a boolean indicating whether to stop.
    """
    if val_metric < best_val_metric:
        best_val_metric = val_metric
        torch.save(model.state_dict(), save_path)
        patience_counter = 0

    else:
        patience_counter += 1

    stop_training = patience_counter >= patience_limit
    return best_val_metric, patience_counter, stop_training
