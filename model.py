from __future__ import annotations

import PIL.Image
import torch
from diffusers import (StableDiffusionAttendAndExcitePipeline,
                       StableDiffusionPipeline)


class Model:
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        model_id = 'CompVis/stable-diffusion-v1-4'
        if self.device.type == 'cuda':
            self.ax_pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
                model_id, torch_dtype=torch.float16)
            self.ax_pipe.to(self.device)
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16)
            self.sd_pipe.to(self.device)
        else:
            self.ax_pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
                model_id)
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(model_id)

    def get_token_table(self, prompt: str):
        tokens = [
            self.ax_pipe.tokenizer.decode(t)
            for t in self.ax_pipe.tokenizer(prompt)['input_ids']
        ]
        tokens = tokens[1:-1]
        return list(enumerate(tokens, start=1))

    def run(
        self,
        prompt: str,
        indices_to_alter_str: str,
        seed: int = 0,
        apply_attend_and_excite: bool = True,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        scale_factor: int = 20,
        thresholds: dict[int, float] = {
            10: 0.5,
            20: 0.8,
        },
        max_iter_to_alter: int = 25,
    ) -> PIL.Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if apply_attend_and_excite:
            try:
                token_indices = list(map(int, indices_to_alter_str.split(',')))
            except Exception:
                raise ValueError('Invalid token indices.')
            out = self.ax_pipe(
                prompt=prompt,
                token_indices=token_indices,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_steps,
                max_iter_to_alter=max_iter_to_alter,
                thresholds=thresholds,
                scale_factor=scale_factor,
            )
        else:
            out = self.sd_pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_steps,
            )
        return out.images[0]
