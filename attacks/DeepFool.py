import torch
import torch.nn as nn
import numpy as np


class DeepFool:

    def __init__(self, model, max_iter=50, overshoot=0.02):
        self.model = model.cuda()
        self.model.eval()
        self.max_iter = max_iter
        self.overshoot = overshoot

    def attack(self, data, labels, device):
        org_pc = data.float().to(device).detach()
        adv_pc = org_pc.clone().detach().transpose(1, 2).contiguous()
        labels = labels.long().to(device)

        batch_size = org_pc.shape[0]
        correct_preds = torch.tensor([True for _ in range(batch_size)])

        w = torch.zeros_like(adv_pc)
        perturbation = torch.zeros_like(adv_pc)

        iterations = 0
        success_num = torch.zeros_like(correct_preds)

        while (True in correct_preds) and (iterations < self.max_iter):
            for idx in range(batch_size):
                if not correct_preds[idx]:
                    continue

                adv_data, early_stop = self.attack_single_pc(adv_pc[idx], labels[idx], device)
                adv_pc[idx] = adv_data

                if early_stop:
                    correct_preds[idx] = False

            iterations += 1
        
        success_num = torch.sum(~correct_preds).item()
        return adv_pc.transpose(1, 2).contiguous().detach().cpu().numpy(), success_num

    def attack_single_pc(self, data, target, device):
        data = data.float().to(device)
        adv_data = data.clone().detach()
        adv_data = torch.unsqueeze(adv_data, 0)
        target = target.long().to(device)

        adv_data.requires_grad_()
        logits = self.model(adv_data)

        if isinstance(logits, tuple):
            logits = logits[0]
        
        pred = torch.argmax(logits)
        if pred != target:
            return torch.squeeze(adv_data), True

        # ws = torch.autograd.functional.jacobian(self.model_func, adv_data)
        ws = self.jacobian(logits, adv_data)
        logits = logits.flatten()
        if isinstance(ws, tuple):
            ws = ws[0]
        ws = torch.squeeze(ws)

        adv_data = adv_data.detach().squeeze()
        f0, w0 = logits[target], ws[target]

        wrong_classes = [i for i in range(len(logits)) if i != target]
        fk = logits[wrong_classes]
        wk = ws[wrong_classes]

        f_prime = fk - f0
        w_prime = wk - w0

        obj = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        l_hat = torch.argmin(obj)

        perturbation = torch.abs(f_prime[l_hat]) * w_prime[l_hat] / (torch.norm(w_prime[l_hat], p=2) ** 2)

        adv_data = adv_data + (1 + self.overshoot) * perturbation
        adv_data = torch.clamp(adv_data, min=0, max=1).detach()

        return adv_data, False
        
        
    def model_func(self, data):
        return self.model(data)
        
    
    def jacobian(self, logits, x):
        grads = []
        
        for idx in range(logits.shape[1]):
            logit = logits[0, idx]
            
            if x.grad is not None:
                x.grad.zero_()
                
            logit.backward(retain_graph=True)
            grads += [x.grad.clone().detach()]
            
        return torch.stack(grads).reshape(*logits.shape, *x.shape)
