import torch
import torch.nn as nn
import torch.nn.functional as F
from grafiti.attention import MAB2


def batch_flatten(x_list, mask):
    """
    Flatten a batch of time series based on a mask.

    Args:
        x_list (List[Tensor]): List of tensors with shape (B, T, C)
        mask (Tensor): Mask tensor of shape (B, T, C)

    Returns:
        List[Tensor]: List of flattened tensors with shape (B, K)
    """
    b, t, d = x_list[0].shape
    m_flat = mask.view(b, t * d)

    observed_counts = m_flat.sum(dim=1)
    k = observed_counts.max().to(torch.int64).item()

    indices = torch.arange(k, device=mask.device).expand(b, k)
    mask_indices = indices < observed_counts.unsqueeze(1)

    y_padded = []
    for x in x_list:
        x_flat = x.reshape(b, t * d)
        observed_values = x_flat[m_flat.bool()]
        y_padded_ = torch.full((b, k), 0, device=mask.device, dtype=x_flat.dtype)
        y_padded_[mask_indices] = observed_values
        y_padded.append(y_padded_)

    return y_padded


def reconstruct_y(
    Y_mask: torch.Tensor, Y_flat: torch.Tensor, mask_f: torch.Tensor
) -> torch.Tensor:
    """
    Reconstructs the original tensor Y from its flattened version Y_flat and the mask Y_mask using vectorized operations.

    Args:
        Y_flat: A tensor of shape (B, K), where B is the batch size and K is the maximum
                number of True values in Y_mask across all instances in the batch.
        Y_mask: A boolean tensor of shape (B, T, D), where B is the batch size, T is the
                first dimension of the original Y, and D is the second dimension of the original Y.
                The True values in Y_mask indicate the positions of the elements that were
                flattened into Y_flat.

    Returns:
        Y_reconstructed: A tensor of shape (B, T, D) representing the reconstructed original tensor Y.
    """
    Y_reconstructed = torch.zeros_like(Y_mask, dtype=Y_flat.dtype)

    # Get the indices of True values in Y_mask
    true_indices = torch.nonzero(
        Y_mask, as_tuple=True
    )  # (batch_indices, flattened_indices)
    Y_reconstructed[true_indices] = Y_flat[mask_f.bool()]
    return Y_reconstructed


def gather(x, inds):
    """
    Gather values from tensor based on indices.

    Args:
        x (Tensor): Tensor of shape (B, P, M)
        inds (Tensor): Indices of shape (B, K')

    Returns:
        Tensor: Gathered tensor of shape (B, K', M)
    """
    return x.gather(1, inds[:, :, None].repeat(1, 1, x.shape[-1]))


class grafiti_(nn.Module):
    """GraFITi model"""

    def __init__(
        self,
        dim: int = 41,
        nkernel: int = 128,
        n_layers: int = 3,
        attn_head: int = 4,
        device: str = "cuda",
    ):
        """initializing grafiti model

        Args:
            dim (int, optional): number of channels. Defaults to 41.
            nkernel (int, optional): latent dimension size. Defaults to 128.
            n_layers (int, optional): number of grafiti layers. Defaults to 3.
            attn_head (int, optional): number of attention heads. Defaults to 4.
            device (str, optional): "cpu" or "cuda. Defaults to "cuda".
        """
        super().__init__()
        self.nkernel = nkernel
        self.nheads = attn_head
        self.device = device
        self.n_layers = n_layers

        self.edge_init = nn.Linear(2, nkernel)
        self.chan_init = nn.Linear(dim, nkernel)
        self.time_init = nn.Linear(1, nkernel)

        self.channel_time_attn = nn.ModuleList(
            [
                MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, attn_head)
                for _ in range(n_layers)
            ]
        )
        self.time_channel_attn = nn.ModuleList(
            [
                MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, attn_head)
                for _ in range(n_layers)
            ]
        )
        self.edge_nn = nn.ModuleList(
            [nn.Linear(3 * nkernel, nkernel) for _ in range(n_layers)]
        )

        self.output = nn.Linear(3 * nkernel, 1)
        self.relu = nn.ReLU()

    def _one_hot_channels(
        self, batch_size: int, num_channels: int, device: torch.device
    ) -> torch.Tensor:
        """Creating onehot encoding of channel ids

        Args:
            batch_size (int): B
            num_channels (int): D
            device (torch.device): GPU or CPU

        Returns:
            torch.Tensor: onehot encoding of channels (B, D, D)
        """
        indices = torch.arange(num_channels, device=device).expand(
            batch_size, num_channels
        )
        return F.one_hot(indices, num_classes=num_channels).float()

    def _build_indices(
        self,
        time_steps: torch.Tensor,  # shape: (B, T, 1)
        num_channels: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds index tensors for time steps and channel IDs.

        Args:
            time_steps (torch.Tensor): Input tensor with shape (B, T, 1)
            num_channels (int): Number of channels (D)
            device (torch.device): CPU or GPU

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - t_inds (torch.Tensor): Time indices of shape (B, T, D)
                - c_inds (torch.Tensor): Channel indices of shape (B, T, D)
        """
        b, t = time_steps.shape[0], time_steps.shape[1]

        # Create time indices (B, T, D)
        t_inds = (
            torch.arange(t, device=device).expand(b, num_channels, -1).permute(0, 2, 1)
        )

        # Create channel indices (B, T, D)
        c_inds = torch.arange(num_channels, device=device).expand(b, t, -1)

        return t_inds, c_inds

    def _create_masks(
        self,
        mk: torch.Tensor,
        t_inds_flat: torch.Tensor,
        c_inds_flat: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Creating masks for time and channel attentions in grafiti

        Args:
            mk (torch.Tensor): flattened mask; (B, K')
            t_inds_flat (torch.Tensor): flattened time indices; (B, K')
            c_inds_flat (torch.Tensor): flattened channel indices: (B, K')
            t (torch.Tensor): time points; (B, T)
            c (torch.Tensor): onhot channel encoding; (B, D, D)
            device (torch.Device): GPU or CPU

        Returns:
            tuple[torch.Tensor, torch.Tensor]
            t_mask: time attn mask (B, T, K')
            c_mask: channel attn mask (B, D, K')
        """
        b, t_len = t.shape[:2]
        num_channels = c.shape[1]
        indices = torch.arange(num_channels, device=device).expand(b, num_channels)
        c_mask = (indices[:, :, None] == c_inds_flat[:, None, :]).float() * mk[
            :, None, :
        ]
        t_seq = torch.arange(t_len, device=t.device)[None, :, None]
        t_mask = (t_inds_flat[:, None, :] == t_seq).float() * mk[:, None, :]

        return t_mask, c_mask

    def _encode_features(
        self,
        u_raw: torch.Tensor,
        t: torch.Tensor,
        c_onehot: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoding edge, time node and channel node features

        Args:
            u_raw (torch.Tensor): input edge feature (B, K', 2)
            t (torch.Tensor): time node feature (B, T, 1)
            c_onehot (torch.Tensor): channel node feature (B, C, C)
            mask (torch.Tensor): input mask (B, K')

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Encoded edge features (B, K', M),
            encoded time features (B, T, M),
            encoded channel features (B, C, M)

        """
        u_encoded = self.relu(self.edge_init(u_raw)) * mask[:, :, None]  # (B, K', M)
        t_encoded = torch.sin(self.time_init(t))  # (B, T, M)
        c_encoded = self.relu(self.chan_init(c_onehot))  # (B, C, M)
        return u_encoded, t_encoded, c_encoded

    def forward(
        self,
        time_points: torch.Tensor,
        values: torch.Tensor,
        obs_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """GraFITi model

        Args:
            time_points: time_points have both observed and target times; Tensor (B, T)
            values: Observed values; Tensor (B, T, D)
            obs_mask: Observed values mask; Tensor (B, T, D)
            target_mask: Target values mask; Tensor (B, T, D)

        Returns:
            yhat: Predictions; Tensor (B, T, D)
        """
        b, _, d = values.shape
        t = time_points.unsqueeze(-1)  # (B, T, 1)
        c_onehot = self._one_hot_channels(b, d, device=t.device)  # (B, D, D)

        t_inds, c_inds = self._build_indices(
            time_points, d, t.device
        )  # t_inds (B, T, D), c_inds (B, T, D)

        mask = obs_mask + target_mask
        mask_bool = mask.bool()  # (B, T, D)

        flattened = batch_flatten(
            [t_inds, values, target_mask, c_inds, mask_bool], mask
        )
        t_inds_f, obs_vals, tgt_mask_f, c_inds_f, mask_f = (
            flattened  # all are of shape (B, K'); K' = K+N; K = Number of queries, N' = Total number of observations
        )

        target_indicator = (1 - mask_f.float()) + tgt_mask_f  # target indicator (B, K')
        edge_input = torch.cat(
            [obs_vals.unsqueeze(-1), target_indicator.unsqueeze(-1)], dim=-1
        )  # edge feature (B, K', 2)

        t_mask, c_mask = self._create_masks(
            mask_f, t_inds_f, c_inds_f, t, c_onehot, t.device
        )  # creating masks for attention for time nodes (B, T, K') and channel nodes (B, C, K') respectively

        edge_emb, t_emb, c_emb = self._encode_features(
            edge_input, t, c_onehot, mask_f
        )  # encoding edge features (B, K', M), time node features (B, T, M), channel node features (B, C, M); M is the embedding dimension

        for i in range(self.n_layers):
            t_gathered = gather(t_emb, t_inds_f)  # (B, K', M)

            c_gathered = gather(c_emb, c_inds_f)  # (B, K', M)

            c_emb = self.channel_time_attn[i](
                c_emb, torch.cat([t_gathered, edge_emb], -1), c_mask
            )  # updating channel embedding (B, C, M)
            t_emb = self.time_channel_attn[i](
                t_emb, torch.cat([c_gathered, edge_emb], -1), t_mask
            )  # updating time embedding (B, T, M)

            edge_update = torch.cat(
                [edge_emb, t_gathered, c_gathered], dim=-1
            )  # (B, K', 3*M)

            edge_emb = (
                self.relu(edge_emb + self.edge_nn[i](edge_update)) * mask_f[:, :, None]
            )  # updating edge embedding (B, K', M)

        t_gathered = gather(t_emb, t_inds_f)  # (B, K', M)
        c_gathered = gather(c_emb, c_inds_f)  # (B, K', M)
        output = self.output(
            torch.cat([edge_emb, t_gathered, c_gathered], dim=-1)
        )  # (B, K', 1)
        yhat = reconstruct_y(
            target_mask, output.squeeze(-1), tgt_mask_f
        )  # graph to ts; (B, T, D)

        return yhat
