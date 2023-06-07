from __future__ import annotations

import sys

import gradio as gr
import PIL.Image
import torch

sys.path.append('Attend-and-Excite')

from config import RunConfig
from pipeline_attend_and_excite import AttendAndExcitePipeline
from run import run_on_prompt
from utils.ptp_utils import AttentionStore


class Model:
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_id = ''
        self.model = None
        self.tokenizer = None

        self.load_model('CompVis/stable-diffusion-v1-4')

    def load_model(self, model_id: str) -> None:
        if model_id == self.model_id:
            return
        self.model = AttendAndExcitePipeline.from_pretrained(
            model_id, revision='249dd2d').to(self.device)
        self.tokenizer = self.model.tokenizer
        self.model_id = model_id

    def get_token_table(self, model_id: str, prompt: str):
        self.load_model(model_id)
        tokens = [
            self.tokenizer.decode(t)
            for t in self.tokenizer(prompt)['input_ids']
        ]
        tokens = tokens[1:-1]
        return list(enumerate(tokens, start=1))

    def run(
        self,
        model_id: str,
        prompt: str,
        indices_to_alter_str: str,
        seed: int,
        apply_attend_and_excite: bool,
        num_steps: int,
        guidance_scale: float,
        scale_factor: int = 20,
        thresholds: dict[int, float] = {
            10: 0.5,
            20: 0.8
        },
        max_iter_to_alter: int = 25,
    ) -> PIL.Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        try:
            indices_to_alter = list(map(int, indices_to_alter_str.split(',')))
        except:
            raise gr.Error('Invalid token indices.')

        self.load_model(model_id)

        controller = AttentionStore()
        config = RunConfig(prompt=prompt,
                           n_inference_steps=num_steps,
                           guidance_scale=guidance_scale,
                           run_standard_sd=not apply_attend_and_excite,
                           scale_factor=scale_factor,
                           thresholds=thresholds,
                           max_iter_to_alter=max_iter_to_alter)
        image = run_on_prompt(model=self.model,
                              prompt=[prompt],
                              controller=controller,
                              token_indices=indices_to_alter,
                              seed=generator,
                              config=config)

        return image
