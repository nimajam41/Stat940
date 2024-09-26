import sys
sys.path.append('/content/drive/MyDrive/stat940project')
from attacks import FGM, IFGM
import torch
import torch.nn as nn
import numpy as np
import math

class PGD(IFGM):

    def __init__(self, model, epsilon):

        super(PGD, self).__init__(model, epsilon)

    def create_start(self, data):
        dimensionality = math.sqrt(data.shape[1] * data.shape[2])
        noise = self.epsilon / dimensionality
        perturbation = torch.rand_like(data) * 2 * noise - noise
        with torch.no_grad():
            return data + perturbation


    def attack(self, data, target, device, targeted=True, signed=False):
        start_point = self.create_start(data)
        return super(PGD, self).attack(start_point, target, device, targeted=targeted, signed=signed)