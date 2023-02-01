#!/usr/bin/env python

from __future__ import annotations

import gradio as gr
import PIL.Image

from model import Model

DESCRIPTION = '''# Attend-and-Excite

This is an unofficial demo for [https://github.com/AttendAndExcite/Attend-and-Excite](https://github.com/AttendAndExcite/Attend-and-Excite).
'''

model = Model()


def process_example(
    prompt: str,
    indices_to_alter_str: str,
    seed: int,
    apply_attend_and_excite: bool,
) -> tuple[list[tuple[int, str]], PIL.Image.Image]:
    model_id = 'CompVis/stable-diffusion-v1-4'
    num_steps = 50
    guidance_scale = 7.5
    return model.run(model_id, prompt, indices_to_alter_str, seed,
                     apply_attend_and_excite, num_steps, guidance_scale)


with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            model_id = gr.Text(label='Model ID',
                               value='CompVis/stable-diffusion-v1-4',
                               visible=False)
            prompt = gr.Text(
                label='Prompt',
                max_lines=1,
                placeholder=
                'A pod of dolphins leaping out of the water in an ocean with a ship on the background'
            )
            with gr.Accordion(label='Check token indices', open=False):
                show_token_indices_button = gr.Button('Show token indices')
                token_indices_table = gr.Dataframe(label='Token indices',
                                                   headers=['Index', 'Token'],
                                                   col_count=2)
            token_indices_str = gr.Text(
                label=
                'Token indices (a comma-separated list indices of the tokens you wish to alter)',
                max_lines=1,
                placeholder='4,16')
            seed = gr.Slider(label='Seed',
                             minimum=0,
                             maximum=100000,
                             value=0,
                             step=1)
            apply_attend_and_excite = gr.Checkbox(
                label='Apply Attend-and-Excite', value=True)
            num_steps = gr.Slider(label='Number of steps',
                                  minimum=0,
                                  maximum=100,
                                  step=1,
                                  value=50)
            guidance_scale = gr.Slider(label='CFG scale',
                                       minimum=0,
                                       maximum=50,
                                       step=0.1,
                                       value=7.5)
            run_button = gr.Button('Generate')
        with gr.Column():
            result = gr.Image(label='Result')

    with gr.Row():
        examples = [
            [
                'A horse and a dog',
                '2,5',
                123,
                True,
            ],
            [
                'A horse and a dog',
                '2,5',
                123,
                False,
            ],
            [
                'A painting of an elephant with glasses',
                '5,7',
                123,
                True,
            ],
            [
                'A painting of an elephant with glasses',
                '5,7',
                123,
                False,
            ],
            [
                'A playful kitten chasing a butterfly in a wildflower meadow',
                '3,6,10',
                123,
                True,
            ],
            [
                'A playful kitten chasing a butterfly in a wildflower meadow',
                '3,6,10',
                123,
                False,
            ],
            [
                'A grizzly bear catching a salmon in a crystal clear river surrounded by a forest',
                '2,6,15',
                123,
                True,
            ],
            [
                'A grizzly bear catching a salmon in a crystal clear river surrounded by a forest',
                '2,6,15',
                123,
                False,
            ],
            [
                'A pod of dolphins leaping out of the water in an ocean with a ship on the background',
                '4,16',
                123,
                True,
            ],
            [
                'A pod of dolphins leaping out of the water in an ocean with a ship on the background',
                '4,16',
                123,
                False,
            ],
        ]
        gr.Examples(examples=examples,
                    inputs=[
                        prompt,
                        token_indices_str,
                        seed,
                        apply_attend_and_excite,
                    ],
                    outputs=[
                        token_indices_table,
                        result,
                    ],
                    fn=process_example,
                    cache_examples=True)

    show_token_indices_button.click(fn=model.get_token_table,
                                    inputs=[
                                        model_id,
                                        prompt,
                                    ],
                                    outputs=token_indices_table)

    inputs = [
        model_id,
        prompt,
        token_indices_str,
        seed,
        apply_attend_and_excite,
        num_steps,
        guidance_scale,
    ]
    outputs = [
        token_indices_table,
        result,
    ]
    prompt.submit(fn=model.run, inputs=inputs, outputs=outputs)
    token_indices_str.submit(fn=model.run, inputs=inputs, outputs=outputs)
    run_button.click(fn=model.run, inputs=inputs, outputs=outputs)

demo.queue(max_size=1).launch(share=False)