#!/usr/bin/env python

from __future__ import annotations

import argparse

import gradio as gr
import numpy as np

from model import Model

TITLE = '# Self-Distilled StyleGAN'
DESCRIPTION = '''This is an unofficial demo for [https://github.com/self-distilled-stylegan/self-distilled-internet-photos](https://github.com/self-distilled-stylegan/self-distilled-internet-photos).

Expected execution time on Hugging Face Spaces: 2s'''
FOOTER = '<img id="visitor-badge" src="https://visitor-badge.glitch.me/badge?page_id=hysts.self-distilled-stylegan" alt="visitor badge" />'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def get_sample_image_url(model_name: str) -> str:
    sample_image_dir = 'https://huggingface.co/spaces/hysts/Self-Distilled-StyleGAN/resolve/main/samples'
    return f'{sample_image_dir}/{model_name}.jpg'


def get_sample_image_markdown(model_name: str) -> str:
    url = get_sample_image_url(model_name)
    size = model_name.split('_')[-1]
    return f'''
    - size: {size}x{size}
    - seed: 0-99
    - truncation: 0.7
    ![sample images]({url})'''


def get_cluster_center_image_url(model_name: str) -> str:
    cluster_center_image_dir = 'https://huggingface.co/spaces/hysts/Self-Distilled-StyleGAN/resolve/main/cluster_center_images'
    return f'{cluster_center_image_dir}/{model_name}.jpg'


def get_cluster_center_image_markdown(model_name: str) -> str:
    url = get_cluster_center_image_url(model_name)
    return f'![cluster center images]({url})'


def update_distance_type(multimodal_truncation: bool) -> dict:
    return gr.Dropdown.update(visible=multimodal_truncation)


def main():
    args = parse_args()

    model = Model(args.device)

    with gr.Blocks(theme=args.theme, css='style.css') as demo:
        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)

        with gr.Tabs():
            with gr.TabItem('App'):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            model_name = gr.Dropdown(
                                model.MODEL_NAMES,
                                value=model.MODEL_NAMES[0],
                                label='Model')
                            seed = gr.Slider(0,
                                             np.iinfo(np.uint32).max,
                                             value=0,
                                             step=1,
                                             label='Seed')
                            psi = gr.Slider(0,
                                            2,
                                            step=0.05,
                                            value=0.7,
                                            label='Truncation psi')
                            multimodal_truncation = gr.Checkbox(
                                label='Multi-modal Truncation', value=True)
                            distance_type = gr.Dropdown([
                                'lpips',
                                'l2',
                            ],
                                                        value='lpips',
                                                        label='Distance Type')
                            run_button = gr.Button('Run')
                    with gr.Column():
                        result = gr.Image(label='Result', elem_id='result')
            with gr.TabItem('Sample Images'):
                with gr.Row():
                    model_name2 = gr.Dropdown(model.MODEL_NAMES,
                                              value=model.MODEL_NAMES[0],
                                              label='Model')
                with gr.Row():
                    text = get_sample_image_markdown(model_name2.value)
                    sample_images = gr.Markdown(text)
            with gr.TabItem('Cluster Center Images'):
                with gr.Row():
                    model_name3 = gr.Dropdown(model.MODEL_NAMES,
                                              value=model.MODEL_NAMES[0],
                                              label='Model')
                with gr.Row():
                    text = get_cluster_center_image_markdown(model_name3.value)
                    cluster_center_images = gr.Markdown(value=text)

        gr.Markdown(FOOTER)

        model_name.change(fn=model.set_model, inputs=model_name, outputs=None)
        multimodal_truncation.change(fn=update_distance_type,
                                     inputs=multimodal_truncation,
                                     outputs=distance_type)
        run_button.click(fn=model.set_model_and_generate_image,
                         inputs=[
                             model_name,
                             seed,
                             psi,
                             multimodal_truncation,
                             distance_type,
                         ],
                         outputs=result)
        model_name2.change(fn=get_sample_image_markdown,
                           inputs=model_name2,
                           outputs=sample_images)
        model_name3.change(fn=get_cluster_center_image_markdown,
                           inputs=model_name3,
                           outputs=cluster_center_images)

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
