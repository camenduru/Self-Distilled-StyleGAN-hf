from __future__ import annotations

import pathlib
import pickle
import sys

import lpips
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

current_dir = pathlib.Path(__file__).parent
submodule_dir = current_dir / 'stylegan3'
sys.path.insert(0, submodule_dir.as_posix())


class LPIPS(lpips.LPIPS):
    @staticmethod
    def preprocess(image: np.ndarray) -> torch.Tensor:
        data = torch.from_numpy(image).float() / 255
        data = data * 2 - 1
        return data.permute(2, 0, 1).unsqueeze(0)

    @torch.inference_mode()
    def compute_features(self, data: torch.Tensor) -> list[torch.Tensor]:
        data = self.scaling_layer(data)
        data = self.net(data)
        return [lpips.normalize_tensor(x) for x in data]

    @torch.inference_mode()
    def compute_distance(self, features0: list[torch.Tensor],
                         features1: list[torch.Tensor]) -> float:
        res = 0
        for lin, x0, x1 in zip(self.lins, features0, features1):
            d = (x0 - x1)**2
            y = lin(d)
            y = lpips.lpips.spatial_average(y)
            res += y.item()
        return res


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
    TRUNCATION_TYPES = [
        'Multimodal (LPIPS)',
        'Multimodal (L2)',
        'Global',
    ]

    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self._download_all_models()
        self._download_all_cluster_centers()
        self._download_all_cluster_center_images()

        self.model_name = self.MODEL_NAMES[0]
        self.model = self._load_model(self.model_name)
        self.cluster_centers = self._load_cluster_centers(self.model_name)
        self.cluster_center_images = self._load_cluster_center_images(
            self.model_name)

        self.lpips = LPIPS()
        self.cluster_center_lpips_feature_dict = self._compute_cluster_center_lpips_features(
        )

    def _load_model(self, model_name: str) -> nn.Module:
        path = hf_hub_download('public-data/Self-Distilled-StyleGAN',
                               f'models/{model_name}_pytorch.pkl')
        with open(path, 'rb') as f:
            model = pickle.load(f)['G_ema']
        model.eval()
        model.to(self.device)
        return model

    def _load_cluster_centers(self, model_name: str) -> torch.Tensor:
        path = hf_hub_download('public-data/Self-Distilled-StyleGAN',
                               f'cluster_centers/{model_name}.npy')
        centers = np.load(path)
        centers = torch.from_numpy(centers).float().to(self.device)
        return centers

    def _load_cluster_center_images(self, model_name: str) -> np.ndarray:
        path = hf_hub_download('public-data/Self-Distilled-StyleGAN',
                               f'cluster_center_images/{model_name}.npy')
        return np.load(path)

    def set_model(self, model_name: str) -> None:
        if model_name == self.model_name:
            return
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.cluster_centers = self._load_cluster_centers(model_name)
        self.cluster_center_images = self._load_cluster_center_images(
            model_name)

    def _download_all_models(self):
        for name in self.MODEL_NAMES:
            self._load_model(name)

    def _download_all_cluster_centers(self):
        for name in self.MODEL_NAMES:
            self._load_cluster_centers(name)

    def _download_all_cluster_center_images(self):
        for name in self.MODEL_NAMES:
            self._load_cluster_center_images(name)

    def generate_z(self, seed: int) -> torch.Tensor:
        seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
        return torch.from_numpy(
            np.random.RandomState(seed).randn(1, self.model.z_dim)).float().to(
                self.device)

    def compute_w(self, z: torch.Tensor) -> torch.Tensor:
        label = torch.zeros((1, self.model.c_dim), device=self.device)
        w = self.model.mapping(z, label)
        return w

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

    def compute_lpips_features(self, image: np.ndarray) -> list[torch.Tensor]:
        data = self.lpips.preprocess(image)
        return self.lpips.compute_features(data)

    def _compute_cluster_center_lpips_features(
            self) -> dict[str, list[list[torch.Tensor]]]:
        res = dict()
        for name in self.MODEL_NAMES:
            images = self._load_cluster_center_images(name)
            res[name] = [
                self.compute_lpips_features(image) for image in images
            ]
        return res

    def compute_distance_to_cluster_centers(
            self, ws: torch.Tensor, distance_type: str) -> list[torch.Tensor]:
        if distance_type == 'l2':
            return self._compute_l2_distance_to_cluster_centers(ws)
        elif distance_type == 'lpips':
            return self._compute_lpips_distance_to_cluster_centers(ws)
        else:
            raise ValueError

    def _compute_l2_distance_to_cluster_centers(
            self, ws: torch.Tensor) -> np.ndarray:
        dist2 = ((self.cluster_centers - ws[0, 0])**2).sum(dim=1)
        return dist2.cpu().numpy()

    def _compute_lpips_distance_to_cluster_centers(
            self, ws: torch.Tensor) -> np.ndarray:
        x = self.synthesize(ws)
        x = self.postprocess(x)[0]
        feat0 = self.compute_lpips_features(x)
        cluster_center_features = self.cluster_center_lpips_feature_dict[
            self.model_name]
        distances = [
            self.lpips.compute_distance(feat0, feat1)
            for feat1 in cluster_center_features
        ]
        return np.asarray(distances)

    def find_nearest_cluster_center(self, ws: torch.Tensor,
                                    distance_type: str) -> int:
        distances = self.compute_distance_to_cluster_centers(ws, distance_type)
        return int(np.argmin(distances))

    def generate_image(self, seed: int, truncation_psi: float,
                       truncation_type: str) -> np.ndarray:
        z = self.generate_z(seed)
        ws = self.compute_w(z)
        if truncation_type == self.TRUNCATION_TYPES[2]:
            w0 = self.model.mapping.w_avg
        else:
            if truncation_type == self.TRUNCATION_TYPES[0]:
                distance_type = 'lpips'
            elif truncation_type == self.TRUNCATION_TYPES[1]:
                distance_type = 'l2'
            else:
                raise ValueError
            cluster_index = self.find_nearest_cluster_center(ws, distance_type)
            w0 = self.cluster_centers[cluster_index]
        new_ws = self.truncate_w(w0, ws, truncation_psi)
        out = self.synthesize(new_ws)
        out = self.postprocess(out)
        return out[0]

    def set_model_and_generate_image(self, model_name: str, seed: int,
                                     truncation_psi: float,
                                     truncation_type: str) -> np.ndarray:
        self.set_model(model_name)
        return self.generate_image(seed, truncation_psi, truncation_type)
