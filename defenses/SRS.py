import torch
import torch.nn as nn
import numpy as np


class SRS:
    def __init__(self, drop_num=500):
        self.drop_num = drop_num

    def defense(self, pc):
        with torch.no_grad():
            B, N = pc.shape[:2]
            rand_idx = [np.random.choice(N, N - self.drop_num, replace=False) for _ in range(B)]
            pc = torch.stack([pc[i][torch.from_numpy(rand_idx[i]).long().to(pc.device)] for i in range(B)])
        return pc
