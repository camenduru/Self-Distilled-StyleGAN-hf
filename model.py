from __future__ import annotations

import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

current_dir = pathlib.Path(__file__).parent
submodule_dir = current_dir / 'stylegan3'
sys.path.insert(0, submodule_dir.as_posix())

HF_TOKEN = os.environ['HF_TOKEN']


class Model:

    MODEL_NAMES = [
        'dogs_1024',
        'elephants_512',
        'horses_256',
        'bicycles_256',
        'lions_512',
        'giraffes_512',
        'parrots_512',
    ]

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self._download_all_models()
        self._download_all_cluster_centers()

        self.model_name = self.MODEL_NAMES[0]
        self.model = self._load_model(self.model_name)
        self.cluster_centers = self._load_cluster_centers(self.model_name)

    def _load_model(self, model_name: str) -> nn.Module:
        path = hf_hub_download('hysts/Self-Distilled-StyleGAN',
                               f'models/{model_name}_pytorch.pkl',
                               use_auth_token=HF_TOKEN)
        with open(path, 'rb') as f:
            model = pickle.load(f)['G_ema']
        model.eval()
        model.to(self.device)
        return model

    def _load_cluster_centers(self, model_name: str) -> torch.Tensor:
        path = hf_hub_download('hysts/Self-Distilled-StyleGAN',
                               f'cluster_centers/{model_name}.npy',
                               use_auth_token=HF_TOKEN)
        centers = np.load(path)
        centers = torch.from_numpy(centers).float().to(self.device)
        return centers

    def set_model(self, model_name: str) -> None:
        if model_name == self.model_name:
            return
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.cluster_centers = self._load_cluster_centers(model_name)

    def _download_all_models(self):
        for name in self.MODEL_NAMES:
            self._load_model(name)

    def _download_all_cluster_centers(self):
        for name in self.MODEL_NAMES:
            self._load_cluster_centers(name)

    def generate_z(self, seed: int) -> torch.Tensor:
        seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
        return torch.from_numpy(
            np.random.RandomState(seed).randn(1, self.model.z_dim)).float().to(
                self.device)

    def compute_w(self, z: torch.Tensor) -> torch.Tensor:
        label = torch.zeros((1, self.model.c_dim), device=self.device)
        w = self.model.mapping(z, label)
        return w

    def find_nearest_cluster_center(self, w: torch.Tensor) -> int:
        # Here, Euclidean distance is used instead of LPIPS distance
        dist2 = ((self.cluster_centers - w)**2).sum(dim=1)
        return torch.argmin(dist2).item()

    @staticmethod
    def truncate_w(w_center: torch.Tensor, w: torch.Tensor,
                   psi: float) -> torch.Tensor:
        if psi == 1:
            return w
        return w_center.lerp(w, psi)

    @torch.inference_mode()
    def synthesize(self, w: torch.Tensor) -> torch.Tensor:
        return self.model.synthesis(w)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(
            torch.uint8)
        return tensor.cpu().numpy()

    def generate_image(self, seed: int, truncation_psi: float,
                       multimodal_truncation: bool) -> np.ndarray:
        z = self.generate_z(seed)
        w = self.compute_w(z)
        if multimodal_truncation:
            cluster_index = self.find_nearest_cluster_center(w[:, 0])
            w0 = self.cluster_centers[cluster_index]
        else:
            w0 = self.model.mapping.w_avg
        new_w = self.truncate_w(w0, w, truncation_psi)
        out = self.synthesize(new_w)
        out = self.postprocess(out)
        return out[0]

    def set_model_and_generate_image(
            self, model_name: str, seed: int, truncation_psi: float,
            multimodal_truncation: bool) -> np.ndarray:
        self.set_model(model_name)
        return self.generate_image(seed, truncation_psi, multimodal_truncation)
