import torch
import torch.nn as nn
import torch.nn.functional as F


class PairNorm(nn.Module):
    def __init__(self, scale: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        mean_norm = x.pow(2).sum(dim=1).mean().sqrt() + self.eps
        return self.scale * x / mean_norm


class FeatureDrop(nn.Module):
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0:
            return x
        mask = (torch.rand(x.size(1), device=x.device) > self.p).float()
        return x * mask


def build_lin_in_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_layers: int,
    dropout: float,
) -> nn.Sequential:
    n_layers = int(n_layers)

    if n_layers not in (1, 2, 3):
        raise ValueError(f"LIN_IN_LAYERS must be 1/2/3, got {n_layers}")

    if n_layers == 1:
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        ]
    elif n_layers == 2:
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        ]
    else:
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        ]

    return nn.Sequential(*layers)


class NAMAT(nn.Module):
    """
    NAMAT: Node-wise Message-Aware Multi-graph Attention Network.

    This model integrates multiple PPI networks using node-wise
    message-aware gating.

    Key components:
      - Input encoder (lin_in MLP) before graph message passing
      - Multi-PPI message propagation
      - Gate input: [h_i || m_{i,k} || log(1 + deg_k(i))]
      - Residual connection in each GCN block
      - PairNorm normalization after residual fusion

    Notes:
      - PPI graphs are treated as unweighted binary graphs.
    """

    def __init__(
        self,
        in_dim: int,
        hids: list[int],
        dropout: float,
        featdrop_p: float = 0.2,
        use_deg_aware: bool = True,
        lin_in_layers: int = 3,
        lin_in_hidden: int = 64,
        lin_in_out_dim: int = 64,
    ):
        super().__init__()
        assert hids is not None and len(hids) >= 1

        self.hids = hids
        self.use_deg_aware = use_deg_aware

        self.featdrop = FeatureDrop(featdrop_p)
        self.drop = nn.Dropout(dropout)

        self.lin_in = build_lin_in_mlp(
            in_dim=in_dim,
            hidden_dim=int(lin_in_hidden),
            out_dim=int(lin_in_out_dim),
            n_layers=int(lin_in_layers),
            dropout=dropout,
        )

        block_in_dims = [int(lin_in_out_dim)] + hids[:-1]
        block_out_dims = hids[:]

        self.lin_msg = nn.ModuleList()
        self.gate_msg = nn.ModuleList()
        self.pnorm = nn.ModuleList()
        self.res_proj = nn.ModuleList()

        for d_in, d_out in zip(block_in_dims, block_out_dims):
            self.lin_msg.append(nn.Linear(d_in, d_out, bias=False))

            self.res_proj.append(
                nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out, bias=False)
            )

            gate_in_dim = d_in + d_out + (1 if use_deg_aware else 0)
            hidden = max(d_out, 32)

            self.gate_msg.append(
                nn.Sequential(
                    nn.Linear(gate_in_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1, bias=True),
                )
            )

            self.pnorm.append(PairNorm(scale=1.0))

        d_last = hids[-1]
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_last, d_last),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_last, 1),
        )

    def _block(
        self,
        H: torch.Tensor,
        A_list: list[torch.Tensor],
        mask_list: list[torch.Tensor] | None,
        logdeg_list: list[torch.Tensor] | None,
        lin_msg: nn.Module,
        gate_mlp: nn.Module,
        temp: float,
        force_uniform: bool = False,
        ppi_dropout_p: float = 0.0,
        mask_deg0: bool = True,
        m_clamp: float | None = 20.0,
    ):
        Z = lin_msg(self.drop(H))

        Mk = [torch.sparse.mm(Ak, Z) for Ak in A_list]
        M = torch.stack(Mk, dim=1)

        if not torch.isfinite(M).all():
            M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

        if m_clamp is not None:
            M = torch.clamp(M, -float(m_clamp), float(m_clamp))

        N, K, _ = M.shape

        if force_uniform:
            if mask_deg0 and (mask_list is not None):
                valid_mask = torch.stack(mask_list, dim=1).to(M.device)
                alpha = valid_mask / valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            else:
                alpha = torch.full((N, K), 1.0 / K, device=M.device)

            fused = torch.sum(alpha.unsqueeze(-1) * M, dim=1)
            return fused, alpha

        H_rep = H.unsqueeze(1).expand(-1, K, -1)

        if self.use_deg_aware:
            if logdeg_list is None:
                raise ValueError("logdeg_list must be provided when use_deg_aware=True")
            ld = torch.stack(logdeg_list, dim=1).to(M.device).unsqueeze(-1)
            gate_in = torch.cat([H_rep, M, ld], dim=2)
        else:
            gate_in = torch.cat([H_rep, M], dim=2)

        scores = gate_mlp(gate_in).squeeze(-1)

        if self.training and ppi_dropout_p > 0:
            drop = torch.rand(K, device=scores.device) < ppi_dropout_p
            if drop.all():
                drop[torch.randint(0, K, (1,), device=scores.device)] = False
            scores[:, drop] = scores[:, drop] - 50.0

        alpha = torch.softmax(scores / max(1e-6, float(temp)), dim=1)

        if mask_deg0 and (mask_list is not None):
            valid_mask = torch.stack(mask_list, dim=1).to(alpha.device)
            alpha = alpha * valid_mask
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)

        alpha = alpha.clamp_min(1e-8)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)

        fused = torch.sum(alpha.unsqueeze(-1) * M, dim=1)
        return fused, alpha

    def forward(
        self,
        X: torch.Tensor,
        A_list: list[torch.Tensor],
        mask_list: list[torch.Tensor] | None = None,
        logdeg_list: list[torch.Tensor] | None = None,
        temp: float = 0.6,
        force_uniform: bool = False,
        ppi_dropout_p: float = 0.0,
        mask_deg0: bool = True,
    ):
        H = self.lin_in(self.featdrop(X))
        alphas = []

        for i in range(len(self.hids)):
            fused, alpha = self._block(
                H,
                A_list,
                mask_list,
                logdeg_list,
                self.lin_msg[i],
                self.gate_msg[i],
                temp=temp,
                force_uniform=force_uniform,
                ppi_dropout_p=ppi_dropout_p,
                mask_deg0=mask_deg0,
            )

            pre = fused + self.res_proj[i](H)
            pre = self.pnorm[i](pre)
            H = F.relu(pre)

            alphas.append(alpha)

        logits = self.head(H).squeeze(-1)
        return logits, alphas
