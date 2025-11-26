import torch
import torch.nn as nn
from h_assert_statements import (
    assert_weight_tensor_fulfills_requirements,
)
from h_utils import (
    compute_loss,
)

def set_gaussian_loss():
    return nn.GaussianNLLLoss(reduction="sum")


def nb_nll_loss(y, n, p, y_mask, type_of_computation="sum"):
    """
    y: true values
    y_mask: whether missing mask is given
    """
    nll = (
        torch.lgamma(n)
        + torch.lgamma(y + 1)
        - torch.lgamma(n + y)
        - n * torch.log(p)
        - y * torch.log(1 - p)
    )
    nll = nll * y_mask
    if type_of_computation == "mean":
        return torch.mean(nll)
    elif type_of_computation == "sum":
        return torch.sum(nll)


def nb_zeroinflated_nll_loss(y, n, p, pi, y_mask, device, type_of_computation="sum"):
    """
    y: true values
    y_mask: whether missing mask is given
    https://stats.idre.ucla.edu/r/dae/zinb/
    """
    idx_yeq0 = y == 0
    idx_yg0 = y > 0

    # substract e^-10 to avoid log(0)
    # pi = torch.clamp(pi, 1e-10, 1-1e-10)
    # p = torch.clamp(p, 1e-10, 1-1e-10)

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = y[idx_yeq0]
    weight_yeq0 = y_mask[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = y[idx_yg0]
    weight_yg0 = y_mask[idx_yg0]

    L_yeq0 = torch.log(pi_yeq0) + torch.log((1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0))
    L_yg0 = (
        torch.log(1 - pi_yg0)
        + torch.lgamma(n_yg0 + yg0)
        - torch.lgamma(yg0 + 1)
        - torch.lgamma(n_yg0)
        + n_yg0 * torch.log(p_yg0)
        + yg0 * torch.log(1 - p_yg0)
    )

    L_yeq0_weighted = L_yeq0 * weight_yeq0
    L_yg0_weighted = L_yg0 * weight_yg0

    if type_of_computation == "mean":
        return -torch.mean(L_yeq0_weighted) - torch.mean(L_yg0_weighted)
    elif type_of_computation == "sum":
        return -torch.sum(L_yeq0_weighted) - torch.sum(L_yg0_weighted)


def compute_the_loss(
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
):
    # Compute the loss, we set those observations, which are nan (or encoded as zero) to zero, so they do not effect the loss.
    if compute_train_error_on_hidden_counter_only:
        weight_tensor_for_loss = compute_weight_matrix_for_only_evaluating_on_left_out(
            hiding_mask, mask
        ).to(device)
    else:
        weight_tensor_for_loss = mask.to(device)
        assert_weight_tensor_fulfills_requirements(
            weight_tensor_for_loss,
        )

    if type_of_loss == "ZINB":
        y_batch = outputs[:, :, :, 0].clone().permute(0, 2, 1)
        loss = nb_zeroinflated_nll_loss(
            y_batch,
            n_train,
            p_train,
            pi_train,
            weight_tensor_for_loss[:, :, :, 0].permute(0, 2, 1),
            device,
        )
        # return(valid_pred_mean, valid_y_batch, valid_pred_var, weight_tensor_for_loss)
    elif type_of_loss == "NB":
        y_batch = outputs[:, :, :, 0].clone().permute(0, 2, 1)
        loss = nb_nll_loss(
            y_batch,
            n_train,
            p_train,
            weight_tensor_for_loss[:, :, :, 0].permute(0, 2, 1),
        )
        # return(y_batch, n_train, p_train, weight_tensor_for_loss)
    elif type_of_loss in {"NLL"}:
        y_batch = outputs[:, :, :, 0].clone().permute(0, 2, 1)
        # Ensure variance is positive
        # Check the validity of the weight tensor (only 0 and 1)
        assert torch.all(
            torch.logical_or(
                weight_tensor_for_loss[:, :, :, 0].permute(0, 2, 1) == 0,
                weight_tensor_for_loss[:, :, :, 0].permute(0, 2, 1) == 1,
            )
        )
        # Create a mask for valid entries (where weight is 1)
        valid_mask = weight_tensor_for_loss[:, :, :, 0].permute(0, 2, 1) == 1
        # Filter the y_batch and pred_mean by the valid mask (keep only valid entries)
        valid_y_batch = y_batch[valid_mask]
        valid_pred_mean = pred_mean[valid_mask]
        valid_pred_var = pred_var[valid_mask]  # Only if you're using pred_var
        # Compute the loss on the valid entries
        loss = nll_loss(valid_pred_mean, valid_y_batch, valid_pred_var)
        # return(valid_pred_mean, valid_y_batch, valid_pred_var, weight_tensor_for_loss)

    elif type_of_loss in {
        "MAE",
        "SMAPE",
        "RMSE",
        "MAPE",
        "MSE",
    }:
        loss = compute_loss(
            X_res,
            outputs[:, :, :, 0],
            weight_tensor_for_loss[:, :, :, 0],
            loss_type=type_of_loss,
        )

    del outputs
    if torch.isnan(loss):
        print("Loss:", loss.item())
        raise ValueError("The loss is nan. The model is not learning.")
    return loss
