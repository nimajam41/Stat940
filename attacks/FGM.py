import torch
import torch.nn as nn
import numpy as np

LEAST_NUM = 10 ** -9

class FGM:

    def __init__(self, model, epsilon):
        self.model = model
        self.model.eval()
        self.adv_loss_func = nn.CrossEntropyLoss()
        self.epsilon = epsilon


    def compute_gradient(self, data, target, device, normalize=True):
        data = data.float().to(device)
        target = target.long().to(device)

        data.requires_grad_()
        logits = self.model(data)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = torch.argmax(logits, dim=1)

        loss = self.adv_loss_func(logits, target).mean()
        loss.backward()

        with torch.no_grad():
            grad = data.grad.detach()
            if normalize:
                norm = torch.norm(grad, dim=(1, 2))
                grad = grad / (norm[:, None, None] + LEAST_NUM)

        return grad, pred


    def attack(self, data, target, device, targeted=True, signed=False):
        org_pc = data.float().to(device).detach()
        grad_pc = org_pc.clone().detach().transpose(1, 2).contiguous()
        target = target.long().to(device)

        grad, pred = self.compute_gradient(grad_pc, target, device)

        with torch.no_grad():
            if signed:
                grad = torch.sign(grad)

            perturbation = grad * self.epsilon
            perturbation = perturbation.transpose(1, 2).contiguous()

            # Targeted Attack
            if targeted:
                adv_data = org_pc - perturbation

            # Untargeted Attack
            else:
                adv_data = org_pc + perturbation


            logits = self.model(adv_data.transpose(1, 2).contiguous())
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)

            if targeted:
                success_num = (pred == target).sum().item()
            else:
                success_num = (pred != target).sum().item()

        print(f'{success_num}/{adv_data.shape[0]} successful attacks')
        torch.cuda.empty_cache()

        return adv_data.detach().cpu().numpy(), success_num
