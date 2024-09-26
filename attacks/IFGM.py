import sys
sys.path.append('/content/drive/MyDrive/stat940project')
from attacks import FGM
import torch
import torch.nn as nn
import numpy as np

class IFGM(FGM):

    def __init__(self, model, epsilon, iterations=10):

        super(IFGM, self).__init__(model, epsilon)
        self.epsilon = epsilon
        self.iterations = iterations
        self.step = self.epsilon / float(self.iterations)

    def attack(self, data, target, device, targeted=True, signed=False):

        org_pc = data.float().to(device).detach()
        grad_pc = org_pc.clone().detach().transpose(1, 2).contiguous()
        target = target.long().to(device)

        for iteration in range(self.iterations):

            grad, pred = self.compute_gradient(grad_pc, target, device)

            with torch.no_grad():
                if signed:
                    grad = torch.sign(grad)

                perturbation = grad * self.step

                if targeted:
                    grad_pc = grad_pc - perturbation

                else:
                    grad_pc = grad_pc + perturbation
                

                if targeted:
                    success_num = (pred == target).sum().item()
                else:
                    success_num = (pred != target).sum().item()

            print(f'{success_num}/{org_pc.shape[0]} successful attacks in iteration ', f'{iteration}')
            torch.cuda.empty_cache()

        with torch.no_grad():
            logits = self.model(grad_pc)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)

            if targeted:
                    success_num = (pred == target).sum().item()
            else:
                success_num = (pred != target).sum().item()

        print(f'{success_num}/{org_pc.shape[0]} Total successful attacks')

        grad_pc = grad_pc.transpose(1, 2).contiguous().detach().cpu().numpy()

        return grad_pc, success_num

