import torch
from torch import nn
import torch.nn.functional as F

class SigmoidLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super().__init__()
        self.adv_temperature = adv_temperature

    def forward(self, p_scores, n_scores):
        if self.adv_temperature:
            weights = F.softmax(self.adv_temperature * n_scores, dim=-1).detach()
            n_scores = weights * n_scores
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()

        return (p_loss + n_loss) / 2

class DiffusionLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred_eps, true_eps):
        return self.mse(
            pred_eps.contiguous().view(pred_eps.size(0), -1),
            true_eps.contiguous().view(true_eps.size(0), -1)
        )

