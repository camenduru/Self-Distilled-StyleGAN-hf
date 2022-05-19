#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pickle
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

sys.path.insert(0, 'stylegan3')

TITLE = 'Self-Distilled StyleGAN'
DESCRIPTION = '''This is an unofficial demo for models provided in https://github.com/self-distilled-stylegan/self-distilled-internet-photos.

Expected execution time on Hugging Face Spaces: 2s
'''
SAMPLE_IMAGE_DIR = 'https://huggingface.co/spaces/hysts/Self-Distilled-StyleGAN/resolve/main/samples'
ARTICLE = f'''## Generated images
- truncation: 0.7
### Dogs
- size: 1024x1024
- seed: 0-99
![Dogs]({SAMPLE_IMAGE_DIR}/dogs.jpg)
### Elephants
- size: 512x512
- seed: 0-99
![Elephants]({SAMPLE_IMAGE_DIR}/elephants.jpg)
### Horses
- size: 256x256
- seed: 0-99
![Horses]({SAMPLE_IMAGE_DIR}/horses.jpg)
### Bicycles
- size: 256x256
- seed: 0-99
![Bicycles]({SAMPLE_IMAGE_DIR}/bicycles.jpg)
### Lions
- size: 512x512
- seed: 0-99
![Lions]({SAMPLE_IMAGE_DIR}/lions.jpg)
### Giraffes
- size: 512x512
- seed: 0-99
![Giraffes]({SAMPLE_IMAGE_DIR}/giraffes.jpg)
### Parrots
- size: 512x512
- seed: 0-99
![Parrots]({SAMPLE_IMAGE_DIR}/parrots.jpg)

<center><img src="https://visitor-badge.glitch.me/badge?page_id=hysts.self-distilled-stylegan" alt="visitor badge"/></center>
'''

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def generate_z(z_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(seed).randn(
        1, z_dim)).to(device).float()


@torch.inference_mode()
def generate_image(model_name: str, seed: int, truncation_psi: float,
                   model_dict: dict[str, nn.Module],
                   device: torch.device) -> np.ndarray:
    model = model_dict[model_name]
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))

    z = generate_z(model.z_dim, seed, device)
    label = torch.zeros([1, model.c_dim], device=device)

    out = model(z, label, truncation_psi=truncation_psi)
    out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return out[0].cpu().numpy()


def load_model(model_name: str, device: torch.device) -> nn.Module:
    path = hf_hub_download('hysts/Self-Distilled-StyleGAN',
                           f'models/{model_name}_pytorch.pkl',
                           use_auth_token=TOKEN)
    with open(path, 'rb') as f:
        model = pickle.load(f)['G_ema']
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.z_dim)).to(device)
        label = torch.zeros([1, model.c_dim], device=device)
        model(z, label)
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)

    model_names = [
        'dogs_1024',
        'elephants_512',
        'horses_256',
        'bicycles_256',
        'lions_512',
        'giraffes_512',
        'parrots_512',
    ]

    model_dict = {name: load_model(name, device) for name in model_names}

    func = functools.partial(generate_image,
                             model_dict=model_dict,
                             device=device)
    func = functools.update_wrapper(func, generate_image)

    gr.Interface(
        func,
        [
            gr.inputs.Radio(
                model_names, type='value', default='dogs_1024', label='Model'),
            gr.inputs.Number(default=0, label='Seed'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
