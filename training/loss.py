import torch
import torch.nn as nn


class DirectionAwareJointLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_reg, pred_dir, target_reg, target_dir):
        l_reg = self.mse(pred_reg, target_reg)
        l_dir = self.bce(pred_dir, target_dir)
        sign_penalty = torch.relu(-pred_reg * target_reg).mean()
        total_loss = l_reg + (self.alpha * l_dir) + (self.beta * sign_penalty)
        return total_loss, l_reg, l_dir
