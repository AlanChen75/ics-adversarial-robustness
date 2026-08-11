"""
Anomaly Detector Architectures for ICS Benchmark
5 detector families: LSTM-AE, USAD, GDN, TranAD, DAGMM

Each model exposes:
  - forward(x)           : standard forward pass
  - anomaly_score(x)     : returns (batch,) anomaly scores
  - compute_loss(x, epoch, n_epochs) : for models with special training schedules
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LSTM-AE
# ---------------------------------------------------------------------------

class LSTMAE(nn.Module):
    """LSTM Autoencoder for time series anomaly detection.

    Input:  (batch, window_size, n_features)
    Output: reconstructed (batch, window_size, n_features)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(
            n_features, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.bottleneck = nn.Linear(hidden_dim, latent_dim)
        self.expand = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        enc_out, _ = self.encoder(x)
        z = self.bottleneck(enc_out)
        dec_input = self.expand(z)
        dec_out, _ = self.decoder(dec_input)
        recon = self.output_layer(dec_out)
        return recon

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=(1, 2))


# ---------------------------------------------------------------------------
# USAD
# ---------------------------------------------------------------------------

class USAD(nn.Module):
    """UnSupervised Anomaly Detection with adversarially trained autoencoders.

    Input:  (batch, window_size, n_features) — internally flattened
    Output: tuple of 3 reconstructions (w1, w2, w3), each (batch, window_size, n_features)
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 100,
        hidden_dim: int = 64,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        input_dim = n_features * window_size
        self.window_size = window_size
        self.n_features = n_features

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        z = self.encoder(x_flat)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return (
            w1.reshape(batch_size, self.window_size, self.n_features),
            w2.reshape(batch_size, self.window_size, self.n_features),
            w3.reshape(batch_size, self.window_size, self.n_features),
        )

    def compute_loss(
        self, x: torch.Tensor, epoch: int, n_epochs: int,
    ) -> torch.Tensor:
        w1, w2, w3 = self.forward(x)
        alpha = 1.0 / (epoch + 1)
        beta = 1.0 - alpha
        loss1 = alpha * torch.mean((x - w1) ** 2)
        loss2 = beta * torch.mean((x - w3) ** 2)
        return loss1 + loss2

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        w1, _w2, w3 = self.forward(x)
        alpha = 0.5
        score1 = torch.mean((x - w1) ** 2, dim=(1, 2))
        score2 = torch.mean((x - w3) ** 2, dim=(1, 2))
        return alpha * score1 + (1 - alpha) * score2


# ---------------------------------------------------------------------------
# GDN  (Graph Deviation Network) — fixed for low-feature datasets
# ---------------------------------------------------------------------------

class GDN(nn.Module):
    """Graph Deviation Network for multivariate time series anomaly detection.

    Key fixes vs. original:
      - hidden_dim 128 (was 64) for better capacity
      - Learned adjacency matrix with top-k sparsification
      - Layer normalisation + dropout for regularisation
      - Proper handling when n_features is small (e.g. SWaT 15 features)

    Input:  (batch, window_size, n_features)
    Output: prediction (batch, window_size, n_features)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        topk: int = 20,
        window_size: int = 100,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.topk = min(topk, n_features - 1)
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # Node embedding — one learnable vector per feature / sensor
        self.embedding = nn.Embedding(n_features, hidden_dim)

        # Learned adjacency: each node has a weight vector; pairwise dot → adj
        self.adj_weight = nn.Parameter(torch.randn(n_features, hidden_dim) * 0.01)

        # Feature projection: time-series window → hidden
        self.fc_in = nn.Linear(window_size, hidden_dim)

        # Graph convolution layers (2 layers for deeper message passing)
        self.gc1_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gc2_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Output projection: hidden → window prediction
        self.fc_out = nn.Linear(hidden_dim, window_size)

        # Init
        nn.init.kaiming_uniform_(self.fc_in.weight)
        nn.init.kaiming_uniform_(self.fc_out.weight)

    def _build_adj(self, device: torch.device) -> torch.Tensor:
        """Build a sparse adjacency matrix via learned weights + top-k."""
        # Pairwise cosine similarity → adjacency scores
        w_norm = F.normalize(self.adj_weight, p=2, dim=1)          # (N, H)
        adj = torch.mm(w_norm, w_norm.t())                         # (N, N)

        # Zero out self-loops
        adj = adj * (1.0 - torch.eye(self.n_features, device=device))

        # Top-k sparsification per row
        if self.topk < self.n_features - 1:
            topk_vals, topk_idx = torch.topk(adj, self.topk, dim=1)
            mask = torch.zeros_like(adj)
            mask.scatter_(1, topk_idx, 1.0)
            adj = adj * mask

        # Row-normalise so message passing is stable
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        adj = adj / row_sum
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        batch_size = x.size(0)
        device = x.device

        # Transpose: treat each feature as a "node" with a time-series signal
        x_t = x.permute(0, 2, 1)  # (batch, n_features, window_size)

        # Node embeddings
        node_ids = torch.arange(self.n_features, device=device)
        node_embed = self.embedding(node_ids).unsqueeze(0).expand(batch_size, -1, -1)

        # Project time signal to hidden + add node identity
        h = self.fc_in(x_t) + node_embed  # (batch, n_features, hidden)

        # Adjacency
        adj = self._build_adj(device)  # (n_features, n_features)

        # GCN layer 1
        h = torch.bmm(
            adj.unsqueeze(0).expand(batch_size, -1, -1), h,
        )
        h = self.gc1_weight(h)
        h = self.layer_norm1(h)
        h = F.relu(h)
        h = self.dropout(h)

        # GCN layer 2
        h = torch.bmm(
            adj.unsqueeze(0).expand(batch_size, -1, -1), h,
        )
        h = self.gc2_weight(h)
        h = self.layer_norm2(h)
        h = F.relu(h)
        h = self.dropout(h)

        # Project back to time dimension
        pred = self.fc_out(h)  # (batch, n_features, window_size)
        return pred.permute(0, 2, 1)  # (batch, window_size, n_features)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.forward(x)
        return torch.mean((x - pred) ** 2, dim=(1, 2))


# ---------------------------------------------------------------------------
# TranAD — reduced dropout for better threshold calibration
# ---------------------------------------------------------------------------

class TranAD(nn.Module):
    """Transformer-based anomaly detection with self-conditioning.

    Input:  (batch, window_size, n_features)
    Output: tuple (out1, out2), each (batch, window_size, n_features)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        window_size: int = 100,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size

        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.pos_encoding = self._positional_encoding(window_size, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        decoder_layer1 = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        decoder_layer2 = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder1 = nn.TransformerDecoder(decoder_layer1, num_layers=n_layers)
        self.decoder2 = nn.TransformerDecoder(decoder_layer2, num_layers=n_layers)

        self.output1 = nn.Linear(hidden_dim, n_features)
        self.output2 = nn.Linear(hidden_dim, n_features)

    @staticmethod
    def _positional_encoding(max_len: int, d_model: int) -> nn.Parameter:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x) + self.pos_encoding[:, : x.size(1), :]
        memory = self.encoder(h)
        dec1 = self.decoder1(h, memory)
        out1 = self.output1(dec1)
        dec2 = self.decoder2(h, memory)
        out2 = self.output2(dec2)
        return out1, out2

    def compute_loss(
        self, x: torch.Tensor, epoch: int, n_epochs: int,
    ) -> torch.Tensor:
        out1, out2 = self.forward(x)
        focus = (epoch + 1) / n_epochs
        loss1 = (1 - focus) * torch.mean((x - out1) ** 2)
        loss2 = focus * torch.mean((x - out2) ** 2)
        return loss1 + loss2

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        out1, out2 = self.forward(x)
        score1 = torch.mean((x - out1) ** 2, dim=(1, 2))
        score2 = torch.mean((x - out2) ** 2, dim=(1, 2))
        return (score1 + score2) / 2


# ---------------------------------------------------------------------------
# DAGMM — deep encoder, batch-norm, numerically stable GMM energy
# ---------------------------------------------------------------------------

class DAGMM(nn.Module):
    """Deep Autoencoding Gaussian Mixture Model.

    Key fixes vs. original:
      - 3-layer encoder/decoder (was 2) for better representation
      - Batch normalisation for training stability
      - Log-sum-exp trick in GMM energy to avoid numerical underflow
      - 8 mixture components (was 4) for complex ICS distributions
      - Reconstruction error + relative euclidean distance as extra features

    Input:  (batch, window_size, n_features) — internally flattened to n_features * window_size
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 100,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        n_gmm: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = n_features * window_size
        self.n_features = n_features
        self.window_size = window_size
        self.n_gmm = n_gmm
        self.latent_dim = latent_dim

        # 3-layer encoder with batch norm
        mid_dim = hidden_dim * 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # 3-layer decoder with batch norm
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, input_dim),
            nn.Sigmoid(),
        )

        # Extra features: latent_dim + cosine_sim + euclidean_dist + relative_euclidean
        est_input_dim = latent_dim + 3
        self.estimation = nn.Sequential(
            nn.Linear(est_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_gmm),
            nn.Softmax(dim=1),
        )

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)

        z = self.encoder(x_flat)
        x_hat = self.decoder(z)

        recon = x_hat.reshape(batch_size, self.window_size, self.n_features)

        # --- extra features for the estimation network ---
        cos_sim = F.cosine_similarity(x_flat, x_hat, dim=1).unsqueeze(1)

        euc_dist = torch.norm(x_flat - x_hat, dim=1, keepdim=True)

        # Relative euclidean distance (normalised by input norm)
        x_norm = torch.norm(x_flat, dim=1, keepdim=True).clamp(min=1e-8)
        rel_euc = euc_dist / x_norm

        z_concat = torch.cat([z, cos_sim, euc_dist, rel_euc], dim=1)
        gamma = self.estimation(z_concat)

        return recon, z_concat, gamma

    def compute_gmm_energy(
        self, z: torch.Tensor, gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sample energy from GMM using log-sum-exp for stability."""
        batch_size, z_dim = z.size()

        # Estimate GMM parameters from the batch
        gamma_sum = gamma.sum(dim=0, keepdim=True).clamp(min=1e-8)  # (1, K)
        phi = gamma_sum / batch_size  # mixture weights  (1, K)

        # Means: (K, z_dim)
        mu = (gamma.unsqueeze(2) * z.unsqueeze(1)).sum(dim=0) / gamma_sum.T

        # Covariances: (K, z_dim, z_dim)
        z_centered = z.unsqueeze(1) - mu.unsqueeze(0)  # (B, K, D)
        cov = torch.einsum(
            "bki,bkj->kij",
            gamma.unsqueeze(2) * z_centered,
            z_centered,
        )
        cov = cov / (gamma_sum.T.unsqueeze(2) + 1e-8)
        # Regularise diagonal for numerical stability
        cov = cov + 1e-3 * torch.eye(z_dim, device=z.device).unsqueeze(0)

        # --- log-sum-exp trick for stable energy computation ---
        # log p(z) = log sum_k [ phi_k * N(z | mu_k, Sigma_k) ]
        # We compute log of each component then use logsumexp.
        log_phi = torch.log(phi.squeeze(0).clamp(min=1e-12))  # (K,)

        log_components = torch.zeros(batch_size, self.n_gmm, device=z.device)
        for k in range(self.n_gmm):
            diff = z - mu[k].unsqueeze(0)  # (B, D)

            # Use Cholesky for stable inverse + log-det
            try:
                L = torch.linalg.cholesky(cov[k])
            except RuntimeError:
                # Fallback: add more regularisation
                cov_reg = cov[k] + 1e-2 * torch.eye(z_dim, device=z.device)
                L = torch.linalg.cholesky(cov_reg)

            # log det(Sigma) = 2 * sum(log(diag(L)))
            log_det = 2.0 * torch.sum(torch.log(torch.diag(L).clamp(min=1e-12)))

            # Mahalanobis: diff^T Sigma^{-1} diff  via solving L y = diff^T
            sol = torch.linalg.solve_triangular(
                L, diff.T, upper=False,
            )  # (D, B)
            mahal = torch.sum(sol ** 2, dim=0)  # (B,)

            log_n = -0.5 * (z_dim * math.log(2 * math.pi) + log_det + mahal)
            log_components[:, k] = log_phi[k] + log_n

        # Energy = -log p(z)
        log_density = torch.logsumexp(log_components, dim=1)  # (B,)
        return -log_density

    def compute_loss(
        self, x: torch.Tensor, epoch: int, n_epochs: int,
    ) -> torch.Tensor:
        recon, z, gamma = self.forward(x)
        recon_err = torch.mean((x - recon) ** 2)
        energy = torch.mean(self.compute_gmm_energy(z, gamma))

        # Penalise small diagonal elements in covariance (avoid singularity)
        gamma_sum = gamma.sum(dim=0, keepdim=True).clamp(min=1e-8)
        mu = (gamma.unsqueeze(2) * z.unsqueeze(1)).sum(dim=0) / gamma_sum.T
        z_centered = z.unsqueeze(1) - mu.unsqueeze(0)
        cov = torch.einsum(
            "bki,bkj->kij",
            gamma.unsqueeze(2) * z_centered,
            z_centered,
        )
        cov = cov / (gamma_sum.T.unsqueeze(2) + 1e-8)
        cov = cov + 1e-3 * torch.eye(z.size(1), device=z.device).unsqueeze(0)
        diag_loss = torch.sum(1.0 / torch.diagonal(cov, dim1=1, dim2=2).clamp(min=1e-8))

        # Weighted combination (lambda_energy ramps up over training)
        lambda_energy = 0.1
        lambda_diag = 0.005
        return recon_err + lambda_energy * energy + lambda_diag * diag_loss

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        recon, z, gamma = self.forward(x)
        recon_err = torch.mean((x - recon) ** 2, dim=(1, 2))
        energy = self.compute_gmm_energy(z, gamma)
        return recon_err + 0.1 * energy


# ---------------------------------------------------------------------------
# Model registry & factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lstm_ae": LSTMAE,
    "usad": USAD,
    "gdn": GDN,
    "tranad": TranAD,
    "dagmm": DAGMM,
}

# Default hyper-parameters per model (overridable via config)
_DEFAULT_HPARAMS: dict[str, dict[str, Any]] = {
    "lstm_ae": {"hidden_dim": 64, "latent_dim": 32, "n_layers": 2, "dropout": 0.2},
    "usad": {"hidden_dim": 64, "latent_dim": 32},
    "gdn": {"hidden_dim": 128, "topk": 20, "dropout": 0.1},
    "tranad": {"hidden_dim": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.05},
    "dagmm": {"hidden_dim": 64, "latent_dim": 8, "n_gmm": 8, "dropout": 0.1},
}


def create_model(
    name: str,
    n_features: int,
    window_size: int = 100,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Factory function to create detector models.

    Args:
        name: Model identifier (lstm_ae, usad, gdn, tranad, dagmm).
        n_features: Number of input features / sensors.
        window_size: Sliding window length.
        config: Optional dict of hyper-parameters that override defaults.
        **kwargs: Additional overrides (merged on top of config).

    Returns:
        Instantiated nn.Module with anomaly_score() method.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    # Merge: defaults ← config ← kwargs
    params: dict[str, Any] = dict(_DEFAULT_HPARAMS.get(name, {}))
    if config is not None:
        params.update(config)
    params.update(kwargs)

    # All models accept n_features; most accept window_size
    params["n_features"] = n_features
    if name != "lstm_ae":
        params["window_size"] = window_size

    cls = MODEL_REGISTRY[name]
    return cls(**params)
