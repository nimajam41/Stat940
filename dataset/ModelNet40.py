import numpy as np
import torch
from torch.utils.data import Dataset
import os


def load_data(path, mode = 'train'):
    BASE_DIR = os.path.dirname(os.path.abspath(''))
    DATA_DIR = os.path.join(BASE_DIR, path)

    if mode == 'train':
        TRAIN_DIR = os.path.join(DATA_DIR, 'train')
        X_file = os.path.join(TRAIN_DIR, 'train_pointclouds.npy')
        y_file = os.path.join(TRAIN_DIR, 'train_labels.npy')

    elif mode == 'test':
        TEST_DIR = os.path.join(DATA_DIR, 'test')
        X_file = os.path.join(TEST_DIR, 'test_pointclouds.npy')
        y_file = os.path.join(TEST_DIR, 'test_labels.npy')

    return np.load(X_file), np.load(y_file).astype('int')


def load_targets(path):
    BASE_DIR = os.path.dirname(os.path.abspath(''))
    DATA_DIR = os.path.join(BASE_DIR, path)

    TEST_DIR = os.path.join(DATA_DIR, 'test')
    y_file = os.path.join(TEST_DIR, 'test_targets.npy')

    return np.load(y_file).astype('int')
    
    
def load_adv_data(path, targeted=False):
    BASE_DIR = os.path.dirname(os.path.abspath(''))
    DATA_DIR = os.path.join(BASE_DIR, path)
    
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    y_file = os.path.join(TEST_DIR, 'test_labels.npy')
    t_file = os.path.join(TEST_DIR, 'target_labels.npy')
    
    if targeted:
        X_file = os.path.join(TEST_DIR, 'adv_test_target.npy')
    
    else:
        X_file = os.path.join(TEST_DIR, 'adv_test_untarget.npy')
    
    return np.load(X_file), np.load(y_file).astype('int'), np.load(t_file).astype('int')


def save_data(array, name, path, mode = 'test'):
    BASE_DIR = os.path.dirname(os.path.abspath(''))
    DATA_DIR = os.path.join(BASE_DIR, path)
    FILE_DIR = os.path.join(DATA_DIR, mode)
    if not os.path.exists(FILE_DIR):
        os.mkdir(FILE_DIR)
    OUTPUT_FILE = os.path.join(FILE_DIR, name + ".npy")
    with open (OUTPUT_FILE, "wb") as f:
        np.save(f, array)
    

class ModelNet40Dataset(Dataset):
    def __init__(self, X, y, transforms):
        self.X = X
        self.y = y
        self.transforms = transforms


    def __len__(self):
        return len(self.y)


    def __getitem__(self, idx):
        x = self.X[idx]
        x = self.transforms(x)
        y = torch.tensor(self.y[idx])
        return x, y



class AdversarialDataset(Dataset):
    def __init__(self, X, y, t, transforms):
        self.X = X
        self.y = y
        self.t = t
        self.transforms = transforms


    def __len__(self):
        return len(self.y)


    def __getitem__(self, idx):
        x = self.X[idx]
        x = self.transforms(x)
        y = torch.tensor(self.y[idx])
        t = torch.tensor(self.t[idx])
        return x, y, t
