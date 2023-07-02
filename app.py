#!/usr/bin/env python

from __future__ import annotations

import pathlib

import gradio as gr
import numpy as np

from model import Model

DESCRIPTION = '# [Self-Distilled StyleGAN](https://github.com/self-distilled-stylegan/self-distilled-internet-photos)'


def get_sample_image_url(name: str) -> str:
    sample_image_dir = 'https://huggingface.co/spaces/hysts/Self-Distilled-StyleGAN/resolve/main/samples'
    return f'{sample_image_dir}/{name}.jpg'


def get_sample_image_markdown(name: str) -> str:
    url = get_sample_image_url(name)
    size = name.split('_')[1]
    truncation_type = '_'.join(name.split('_')[2:])
    return f'''
    - size: {size}x{size}
    - seed: 0-99
    - truncation: 0.7
    - truncation type: {truncation_type}
    ![sample images]({url})'''


def get_cluster_center_image_url(model_name: str) -> str:
    cluster_center_image_dir = 'https://huggingface.co/spaces/hysts/Self-Distilled-StyleGAN/resolve/main/cluster_center_images'
    return f'{cluster_center_image_dir}/{model_name}.jpg'


def get_cluster_center_image_markdown(model_name: str) -> str:
    url = get_cluster_center_image_url(model_name)
    return f'![cluster center images]({url})'


model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.TabItem('App'):
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        model_name = gr.Dropdown(label='Model',
                                                 choices=model.MODEL_NAMES,
                                                 value=model.MODEL_NAMES[0])
                        seed = gr.Slider(label='Seed',
                                         minimum=0,
                                         maximum=np.iinfo(np.uint32).max,
                                         step=1,
                                         value=0)
                        psi = gr.Slider(label='Truncation psi',
                                        minimum=0,
                                        maximum=2,
                                        step=0.05,
                                        value=0.7)
                        truncation_type = gr.Dropdown(
                            label='Truncation Type',
                            choices=model.TRUNCATION_TYPES,
                            value=model.TRUNCATION_TYPES[0])
                        run_button = gr.Button('Run')
                with gr.Column():
                    result = gr.Image(label='Result', elem_id='result')

        with gr.TabItem('Sample Images'):
            with gr.Row():
                paths = sorted(pathlib.Path('samples').glob('*'))
                names = [path.stem for path in paths]
                model_name2 = gr.Dropdown(label='Type',
                                          choices=names,
                                          value='dogs_1024_multimodal_lpips')
            with gr.Row():
                text = get_sample_image_markdown(model_name2.value)
                sample_images = gr.Markdown(text)

        with gr.TabItem('Cluster Center Images'):
            with gr.Row():
                model_name3 = gr.Dropdown(label='Model',
                                          choices=model.MODEL_NAMES,
                                          value=model.MODEL_NAMES[0])
            with gr.Row():
                text = get_cluster_center_image_markdown(model_name3.value)
                cluster_center_images = gr.Markdown(value=text)

    model_name.change(fn=model.set_model, inputs=model_name)
    run_button.click(fn=model.set_model_and_generate_image,
                     inputs=[
                         model_name,
                         seed,
                         psi,
                         truncation_type,
                     ],
                     outputs=result)
    model_name2.change(fn=get_sample_image_markdown,
                       inputs=model_name2,
                       outputs=sample_images)
    model_name3.change(fn=get_cluster_center_image_markdown,
                       inputs=model_name3,
                       outputs=cluster_center_images)

demo.queue(max_size=10).launch()
