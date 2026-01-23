import torch
import torch.nn as nn
from get_args import get_config
from encoder import DrugEncoder

cfg = get_config()
dataset_name = cfg.dataset_name


class DiffusionDynamicRESCALLowRank(nn.Module):
    def __init__(self, rel_total, dim, rank=16, noise=0.1):
        super().__init__()
        self.rel_total = rel_total
        self.dim = dim
        self.rank = rank
        self.noise = noise

        # diffusion schedule
        self.T = 10
        beta = torch.linspace(1e-4, 0.01, self.T)
        alpha = 1.0 - beta
        self.register_buffer("alpha_bar", torch.cumprod(alpha, dim=0))

        # base interaction rel
        self.R_base = nn.Embedding(rel_total, dim * dim)
        nn.init.xavier_uniform_(self.R_base.weight)

        self.U_net = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim * rank)
        )
        self.V_net = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim * rank)
        )

        self.eps_net = nn.Sequential(
            nn.Linear(dim * dim + 2 * dim + 1, dim * dim),
            nn.ReLU(),
            nn.Linear(dim * dim, dim * dim)
        )

    def forward(self, z_h, z_t, rel, return_loss=False):
        B = z_h.size(0)

        concat_ht = torch.cat([z_h, z_t], dim=-1)
        U = self.U_net(concat_ht).view(B, self.dim, self.rank)
        V = self.V_net(concat_ht).view(B, self.dim, self.rank)

        adaptive_R = torch.bmm(U, V.transpose(1, 2))

        if 'drugbank' in dataset_name:
            R_base = self.R_base(rel).view(B, self.dim, self.dim)
            R = R_base + adaptive_R
        else:
            R = adaptive_R

        score = torch.sum(
            z_h.unsqueeze(1) @ R @ z_t.unsqueeze(2),
            dim=[1, 2]
        )

        if not return_loss:
            return score

        # =============== Diffusion on ΔR ===============
        adaptive_R_flat = adaptive_R.view(B, -1)

        tau = torch.randint(0, self.T, (B,), device=z_h.device)
        eps_true = torch.randn_like(adaptive_R_flat)

        a_t = self.alpha_bar[tau].unsqueeze(1)
        x_noisy = torch.sqrt(a_t) * adaptive_R_flat + torch.sqrt(1.0 - a_t) * eps_true

        tau_embed = tau.float().unsqueeze(1) / self.T

        eps_pred = self.eps_net(
            torch.cat([x_noisy, z_h, z_t, tau_embed], dim=-1)
        )

        return score, eps_pred, eps_true


class DiffusionDDIModel(nn.Module):
    def __init__(self, encoder_cfg, head_cfg):
        super().__init__()
        self.encoder = DrugEncoder(**encoder_cfg)
        self.head = DiffusionDynamicRESCALLowRank(**head_cfg)

    def forward(self, triples, return_loss=False):
        z_h, z_t, rels = self.encoder.encode_pair(triples)

        if not return_loss:
            return self.head(z_h, z_t, rels, return_loss=False)

        score, eps_pred, eps_true = self.head(
            z_h, z_t, rels, return_loss=True
        )
        return score, eps_pred, eps_true
