import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch import optim

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """

    def __init__(
        self, in_channels, out_channels, orders, dropout_rate, activation="relu"
    ):
        # logging.info(
        #     f"Initializing D_GCN with in_channels={in_channels}, out_channels={out_channels}, orders={orders}, dropout_rate={dropout_rate}, activation={activation}..."
        # )
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        :param dropout_rate: Dropout rate to be applied after the main linear transformation.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(
            torch.FloatTensor(in_channels * self.num_matrices, out_channels)
        )
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()
        # logging.info("D_GCN initialized.")

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1.0 / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        # logging.debug("Performing forward pass in D_GCN...")
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0]  # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length
        supports = [A_q, A_h]

        x0 = X.permute(1, 2, 0)  # (num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])
        x0 = x0.to(torch.float32)
        x = torch.unsqueeze(x0, 0)  # add new dimension in position 0
        for support in supports:
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(
            x, shape=[self.num_matrices, num_node, input_size, batch_size]
        )
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(
            x, shape=[batch_size, num_node, input_size * self.num_matrices]
        )
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "linear":
            pass
        else:
            raise ValueError(f"Activation {self.activation} not supported.")

        # Apply dropout
        x = self.dropout(x)
        # logging.debug("Forward pass in D_GCN completed.")
        return x


# Define the NB class first, not mixture version
############################################################################################################
class NBNorm_ZeroInflated(nn.Module):
    def __init__(self, c_in, c_out):
        super(NBNorm_ZeroInflated, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), bias=True
        )

        self.p_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), bias=True
        )

        self.pi_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), bias=True
        )

        self.out_dim = c_out  # output horizon

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        (B, _, N, _) = x.shape  # B: batch_size; N: input nodes
        n = self.n_conv(x).squeeze_(-1)
        p = self.p_conv(x).squeeze_(-1)
        pi = self.pi_conv(x).squeeze_(-1)

        # Reshape
        n = n.view([B, self.out_dim, N])
        p = p.view([B, self.out_dim, N])
        pi = pi.view([B, self.out_dim, N])

        # Ensure n is positive and p between 0 and 1
        n = F.softplus(n)  # Some parameters can be tuned here
        p = F.sigmoid(p)
        pi = F.sigmoid(pi)
        return n.permute([0, 2, 1]), p.permute([0, 2, 1]), pi.permute([0, 2, 1])


class NBNorm(nn.Module):
    def __init__(self, c_in, c_out):
        super(NBNorm, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), bias=True
        )

        self.p_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), bias=True
        )
        self.out_dim = c_out  # output horizon

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        (B, _, N, _) = x.shape  # B: batch_size; N: input nodes
        n = self.n_conv(x).squeeze_(-1)
        p = self.p_conv(x).squeeze_(-1)

        # Reshape
        n = n.view([B, self.out_dim, N])
        p = p.view([B, self.out_dim, N])

        # Ensure n is positive and p between 0 and 1
        n = F.softplus(n)  # Some parameters can be tuned here
        p = F.sigmoid(p)
        return n.permute([0, 2, 1]), p.permute([0, 2, 1])


############################################################################################################
class IGNNK(nn.Module):
    """
      GNN on ST datasets to reconstruct the datasets
     x_s
      |GNN_3
     H_2 + H_1
      |GNN_2
     H_1
      |GNN_1
    x^y_m
    """

    def __init__(self, h, z, k, num_features, dropout_rate, activation):
        super(IGNNK, self).__init__()
        self.time_dimension = h
        self.hidden_dimension = z  # same number of nodes for both layers
        self.order = k  # for convolution
        self.num_features = num_features
        self.activation = activation

        self.GNN1 = D_GCN(
            self.num_features,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )  #'bounded_linear')#'linear')#'sigmoid')#'linear')#'sigmoid')#
        self.GNN2 = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN3 = D_GCN(
            self.hidden_dimension,
            self.time_dimension,
            self.order,
            dropout_rate=dropout_rate,
            activation=self.activation,
        )  #'bounded_linear')#'linear')#'sigmoid')#'linear')#'sigmoid')#

    def forward(self, X, A_q, A_h, A_q_ignored, A_h_ignored):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes, num_features)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X_S = X.permute(0, 2, 1)  # to correct the input dims
        X_s1 = self.GNN1(X_S, A_q, A_h)  # + X_S
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1  # num_nodes, rank
        X_s3 = self.GNN3(X_s2, A_q, A_h)  # + X_s2
        X_res = X_s3.permute(0, 2, 1)
        return X_res


class GNNUI_MEANBASEDERROR(nn.Module):
    """
      GNN on ST datasets to reconstruct the datasets
     x_s
      |GNN_3
     H_2 + H_1
      |GNN_2
     H_1
      |GNN_1
    x^y_m
    """

    def __init__(self, h, z, k, num_features, dropout_rate, activation):
        super(GNNUI_MEANBASEDERROR, self).__init__()
        self.time_dimension = h
        self.hidden_dimension = z  # same number of nodes for both layers
        self.order = k  # for convolution
        self.num_features = num_features
        self.activation = activation

        self.GNN1 = D_GCN(
            self.num_features,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN2 = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN3 = D_GCN(
            self.hidden_dimension,
            self.time_dimension,
            self.order,
            dropout_rate=dropout_rate,
            activation=self.activation,
        )

        self.ln1 = nn.LayerNorm(self.hidden_dimension)
        self.ln2 = nn.LayerNorm(self.hidden_dimension)

    def forward(self, X, A_q, A_h, A_q_ignored, A_h_ignored):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes, num_features)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X_S = X.permute(0, 2, 1)
        X_s1 = self.GNN1(X_S, A_q, A_h)
        X_s1 = self.ln1(X_s1)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1
        X_s2 = self.ln2(X_s2)
        X_s3 = self.GNN3(X_s2, A_q, A_h)
        X_res = X_s3.permute(0, 2, 1)
        return X_res

    ############################################################################################################


class GNNUI_NLL(nn.Module):
    """
      GNN on ST datasets to reconstruct the datasets
     x_s
      |GNN_3
     H_2 + H_1
      |GNN_2
     H_1
      |GNN_1
    x^y_m
    """

    def __init__(self, h, z, k, num_features, dropout_rate, activation):
        super(GNNUI_NLL, self).__init__()
        self.time_dimension = h
        self.hidden_dimension = z  # same number of nodes for both layers
        self.order = k  # for convolution
        self.num_features = num_features
        self.activation = activation

        self.GNN1 = D_GCN(
            self.num_features,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN2 = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN3 = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
            activation=self.activation,
        )

        # Add  normalization layers
        self.ln1 = nn.LayerNorm(self.hidden_dimension)
        self.ln2 = nn.LayerNorm(self.hidden_dimension)
        self.ln3 = nn.LayerNorm(self.hidden_dimension)

        # Separate linear layers for mean and variance
        # self.mu_layer = nn.Linear(self.hidden_dimension, self.time_dimension)
        # self.log_var_layer = nn.Linear(self.hidden_dimension, self.time_dimension)
        self.output_layer = nn.Linear(self.hidden_dimension, self.time_dimension * 2)
        self.output_layer.bias.data[self.time_dimension :].fill_(-2.0)

    def forward(self, X, A_q, A_h, A_q_ignored, A_h_ignored):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes, num_features)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Mean and variance of reconstructed X
        """
        X_S = X.permute(0, 2, 1)  # Reshape input

        X_s1 = self.GNN1(X_S, A_q, A_h)
        X_s1 = self.ln1(X_s1)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1  # Residual connection
        X_s2 = self.ln2(X_s2)
        X_s3 = self.GNN3(X_s2, A_q, A_h)
        X_s3 = self.ln3(X_s3)

        # Compute mean and variance
        output = self.output_layer(X_s3)
        mu, log_var = torch.chunk(output, 2, dim=-1)  # Split into two parts

        var = torch.softmax(log_var, dim=-1)  # Using softmax instead of exp

        return mu, var  # Reshape back to (batch_size, num_timesteps, num_nodes)


##############


class GNNUI_ZINB(nn.Module):
    """
      GNN on ST datasets to reconstruct the datasets
     x_s
      |GNN_3
     H_2 + H_1
      |GNN_2
     H_1
      |GNN_1
    x^y_m
    """

    def __init__(self, h, z, k, num_features, dropout_rate, activation):
        super(GNNUI_ZINB, self).__init__()
        self.time_dimension = h
        self.hidden_dimension = z  # same number of nodes for both layers
        self.order = k  # for convolution
        self.num_features = num_features
        self.activation = activation

        self.GNN1 = D_GCN(
            self.num_features,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )  #'bounded_linear')#'linear')#'sigmoid')#'linear')#'sigmoid')#
        self.GNN2 = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN3 = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
            activation=self.activation,
        )  #'bounded_linear')#'linear')#'sigmoid')#'linear')#'sigmoid')#

        # Add batch normalization layers
        self.ln1 = nn.LayerNorm(self.hidden_dimension)
        self.ln2 = nn.LayerNorm(self.hidden_dimension)
        self.ln3 = nn.LayerNorm(self.hidden_dimension)

        self.SNB = NBNorm_ZeroInflated(self.hidden_dimension, self.time_dimension)

    def forward(self, X, A_q, A_h, A_q_ignored, A_h_ignored):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes, num_features)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X_S = X.permute(0, 2, 1)
        X_s1 = self.GNN1(X_S, A_q, A_h)
        X_s1 = self.ln1(X_s1)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1
        X_s2 = self.ln2(X_s2)
        X_s3 = self.GNN3(X_s2, A_q, A_h)
        X_s3 = self.ln3(X_s3)
        _b, _n, _hs = X_s3.shape
        n_s_nb, p_s_nb, pi_s_nb = self.SNB(X_s3.view(_b, _n, _hs, 1))
        n_res = n_s_nb
        p_res = p_s_nb
        pi_res = pi_s_nb

        return n_res, p_res, pi_res


class GNNUI_ZINB_twoA(nn.Module):
    """
    Updated GNNUI_ZINB to use binary and distance adjacency matrices.
    """

    def __init__(self, h, z, k, num_features, dropout_rate, activation):
        super(GNNUI_ZINB_twoA, self).__init__()
        self.time_dimension = h
        self.hidden_dimension = z  # Hidden dimension size
        self.order = k  # For convolution
        self.num_features = num_features
        self.activation = activation

        # GNN layers for Binary adjacency matrices
        self.GNN1_binary = D_GCN(
            self.num_features,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN2_binary = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN3_binary = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
            activation=self.activation,
        )
        self.GNN1_distance = D_GCN(
            self.num_features,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN2_distance = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN3_distance = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
            activation=self.activation,
        )

        self.ln1_binary = nn.LayerNorm(self.hidden_dimension)
        self.ln2_binary = nn.LayerNorm(self.hidden_dimension)
        self.ln3_binary = nn.LayerNorm(self.hidden_dimension)
        self.ln1_distance = nn.LayerNorm(self.hidden_dimension)
        self.ln2_distance = nn.LayerNorm(self.hidden_dimension)
        self.ln3_distance = nn.LayerNorm(self.hidden_dimension)

        # Fully connected layer after concatenation
        self.fc = nn.Linear(2 * self.hidden_dimension, self.hidden_dimension)

        # Final prediction layer (ZINB)
        self.SNB = NBNorm_ZeroInflated(self.hidden_dimension, self.time_dimension)

    def forward(self, X, A_q_binary, A_h_binary, A_q_distance, A_h_distance):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes, num_features)
        :param A_q_binary: Forward binary random walk matrix (num_nodes, num_nodes)
        :param A_h_binary: Backward binary random walk matrix (num_nodes, num_nodes)
        :param A_q_distance: Forward distance random walk matrix (num_nodes, num_nodes)
        :param A_h_distance: Backward distance random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        # Permute input for compatibility
        X_S = X.permute(0, 2, 1)  # Shape: (batch_size, num_nodes, num_features)

        # Graph convolutions for Binary adjacency matrices
        X_s1_binary = self.GNN1_binary(X_S, A_q_binary, A_h_binary)  # + X_S
        X_s1_binary = self.ln1_binary(X_s1_binary)
        X_s2_binary = (
            self.GNN2_binary(X_s1_binary, A_q_binary, A_h_binary) + X_s1_binary
        )  # num_nodes, rank
        X_s2_binary = self.ln2_binary(X_s2_binary)
        X_s3_binary = self.GNN3_binary(X_s2_binary, A_q_binary, A_h_binary)  # + X_s2
        X_s3_binary = self.ln3_binary(X_s3_binary)

        # Graph convolutions for Distance adjacency matrices
        X_s1_distance = self.GNN1_distance(X_S, A_q_distance, A_h_distance)  # + X_S
        X_s1_distance = self.ln1_distance(X_s1_distance)
        X_s2_distance = (
            self.GNN2_distance(X_s1_distance, A_q_distance, A_h_distance)
            + X_s1_distance
        )  # num_nodes, rank
        X_s2_distance = self.ln2_distance(X_s2_distance)
        X_s3_distance = self.GNN3_distance(
            X_s2_distance, A_q_distance, A_h_distance
        )  # + X_s2
        X_s3_distance = self.ln3_distance(X_s3_distance)

        # Concatenate all embeddings along the feature dimension
        H_concat = torch.cat(
            [X_s3_binary, X_s3_distance], dim=-1
        )  # Shape: (batch_size, num_nodes, 4 * hidden_dimension)

        # Fully connected layer
        H_fc = self.fc(H_concat)  # Shape: (batch_size, num_nodes, hidden_dimension)

        # Pass through final ZINB layer
        _b, _n, _hs = H_fc.shape
        n_s_nb, p_s_nb, pi_s_nb = self.SNB(H_fc.view(_b, _n, _hs, 1))

        return n_s_nb, p_s_nb, pi_s_nb


##############
class GNNUI_NB(nn.Module):
    """
      GNN on ST datasets to reconstruct the datasets
     x_s
      |GNN_3
     H_2 + H_1
      |GNN_2
     H_1
      |GNN_1
    x^y_m
    """

    def __init__(self, h, z, k, num_features, dropout_rate, activation):
        super(GNNUI_NB, self).__init__()
        self.time_dimension = h
        self.hidden_dimension = z  # same number of nodes for both layers
        self.order = k  # for convolution
        self.num_features = num_features
        self.activation = activation

        self.GNN1 = D_GCN(
            self.num_features,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN2 = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
        )
        self.GNN3 = D_GCN(
            self.hidden_dimension,
            self.hidden_dimension,
            self.order,
            dropout_rate=dropout_rate,
            activation=self.activation,
        )

        self.ln1 = nn.LayerNorm(self.hidden_dimension)
        self.ln2 = nn.LayerNorm(self.hidden_dimension)
        self.ln3 = nn.LayerNorm(self.hidden_dimension)

        self.SNB = NBNorm(self.hidden_dimension, self.time_dimension)

    def forward(self, X, A_q, A_h, A_q_ignored, A_h_ignored):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes, num_features)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X_S = X.permute(0, 2, 1)  # to correct the input dims # Silke
        X_s1 = self.GNN1(X_S, A_q, A_h)  # + X_S
        X_s1 = self.ln1(X_s1)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1  # num_nodes, rank
        X_s2 = self.ln2(X_s2)
        X_s3 = self.GNN3(X_s2, A_q, A_h)  # + X_s2
        X_s3 = self.ln3(X_s3)
        _b, _n, _hs = X_s3.shape
        n_s_nb, p_s_nb = self.SNB(X_s3.view(_b, _n, _hs, 1))
        n_res = n_s_nb
        p_res = p_s_nb

        return n_res, p_res


def initialize_model(
    type_of_loss: str,
    do_IGNNK: bool,
    type_of_adjacency_matrix: str,
    include_node_features_if_observation_are_masked_or_missing: bool,
    include_node_features: bool,
    training_set: np.ndarray,
    h: int,
    z: int,
    K: int,
    overfitting_drop_out_rate: float,
    ACTIVATION_FUNCTION_LAST_LAYER: str,
    learning_rate: float,
    weight_decay: float,
    schedule_learning_rate: bool,
    device: torch.device,
):
    """
    Initialize the GNN model, optimizer, and scheduler.

    Args:
        type_of_loss (str): Type of loss function (e.g., "ZINB", "NB", "NLL").
        do_IGNNK (bool): Whether to use IGNNK.
        type_of_adjacency_matrix (str): Type of adjacency matrix.
        include_node_features_if_observation_are_masked_or_missing (bool): Whether to include a feature indicating masked observations.
        include_node_features (bool): Whether to include node features.
        training_set (np.ndarray): Training dataset.
        h (int): Time horizon.
        z (int): Hidden dimension size.
        K (int): Number of diffusion steps.
        overfitting_drop_out_rate (float): Dropout rate to prevent overfitting.
        ACTIVATION_FUNCTION_LAST_LAYER (str): Activation function for the last layer.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        schedule_learning_rate (bool): Whether to schedule the learning rate.
        device (torch.device): Device to use for training.

    Returns:
        tuple: Initialized model, optimizer, and scheduler (if applicable).
    """
    use_twoA = "_and_" in type_of_adjacency_matrix

    layer_mapping = {
        ("ZINB", False): "DGCN_ZINB",
        ("NB", False): "DGCN_NB",
        ("ZINB", True): "DGCN_ZINB_twoA",
        ("NLL", False): "DGCN_NLL",
    }

    if (
        type_of_loss
        in {
            "MSE",
        }
        and do_IGNNK
    ):
        layer_type_temp = "DGCN"

    elif (
        type_of_loss
        in {
            "MAE",
            "RMSE",
            "MSE",
        }
        and do_IGNNK == False
    ):
        layer_type_temp = "DGCN_MEANERROR"

    elif type_of_loss in {"ZINB", "NB", "NLL"}:
        layer_type_temp = layer_mapping.get((type_of_loss, use_twoA))

    model_classes = {
        "DGCN": IGNNK,
        "DGCN_ZINB": GNNUI_ZINB,
        "DGCN_ZINB_twoA": GNNUI_ZINB_twoA,
        "DGCN_NB": GNNUI_NB,
        "DGCN_NLL": GNNUI_NLL,
        "DGCN_MEANERROR": GNNUI_MEANBASEDERROR,
    }

    ###########################
    # Node feature if observation is missing
    ###########################
    # Possibly add a further value to the input data, which indicates if an observation is hidden.
    if include_node_features_if_observation_are_masked_or_missing:
        shape_parameter = 2
    else:
        shape_parameter = 0

    ###########################
    # Initiate model
    ###########################
    GNN_model = model_classes[layer_type_temp](
        h,
        z,
        K,
        (
            (training_set.shape[2] + shape_parameter) * h
            if include_node_features
            else h
        ),  # we add +1 as we later include a bianry feature indicating hidden observations
        overfitting_drop_out_rate,
        ACTIVATION_FUNCTION_LAST_LAYER,
    )

    optimizer = optim.Adam(
        GNN_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    if schedule_learning_rate:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    GNN_model = GNN_model.to(device)

    return GNN_model, optimizer, scheduler
