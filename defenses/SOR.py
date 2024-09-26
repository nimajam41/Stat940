# Thanks to authors of https://github.com/RyanHangZhou/tensorflow-DUP-Net for open-sourcing their code (TensorFlow)

import torch
import torch.nn as nn
import numpy as np


class SOR:
    def __init__(self, k=2, alpha=1.1):
        self.k = k
        self.alpha = alpha

    def defense(self, x):
        with torch.no_grad():
            pc = x.clone().detach().double()
            B = pc.shape[0]

            pc = pc.transpose(2, 1)
            inner = -2. * torch.matmul(pc.transpose(2, 1), pc)
            xx = torch.sum(pc ** 2, dim=1, keepdim=True)
            dist = xx + inner + xx.transpose(2, 1)
            assert dist.min().item() >= -1e-6

            neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)
            value = -(neg_value[..., 1:])
            value = torch.mean(value, dim=-1)
            mean = torch.mean(value, dim=-1)
            std = torch.std(value, dim=-1)

            threshold = mean + self.alpha * std
            bool_mask = (value <= threshold[:, None])
            out_pc = [x[i][bool_mask[i]] for i in range(B)]
            
            return out_pc
