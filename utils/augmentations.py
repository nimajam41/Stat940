import torch
import numpy as np


class Normalize(object):
    def __call__(self, verts):
        normalized_points = verts - np.mean(verts, axis=0)
        max_norm = np.max(np.linalg.norm(normalized_points, axis=1))

        normalized_points = normalized_points / max_norm

        return normalized_points


class Translate(object):
    def __call__(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud


class RandomRotate(object):
    def __call__(self, verts):
        theta = 2 * np.random.uniform() * np.pi
        rotation_mat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        rotated = np.matmul(verts, rotation_mat)

        return rotated


class RandomNoise(object):
    def __call__(self, verts):
        noise = np.random.normal(0, 0.01, verts.shape)
        noise = np.clip(noise, -0.05, 0.05)
        return verts + noise


class Shuffle(object):
    def __call__(self, verts):
        np.random.shuffle(verts)
        return verts


class ToTensor(object):
    def __call__(self, verts):
        return torch.from_numpy(verts)
