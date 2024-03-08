# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
import time
from cog import BasePredictor, Input, Path
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from src.scheduler_perflow import PeRFlowScheduler

MODELS = [
    "hansyan/perflow-sd15-dreamshaper",
    "hansyan/perflow-sd15-realisticVisionV51",
    "hansyan/perflow-sd15-disney",
]
MODEL_URLS = {
    f"hansyan/perflow-sd15-{m}": f"https://weights.replicate.delivery/default/piecewise-rectified-flow/perflow-sd15-{m}.tar"
    for m in ["dreamshaper", "realisticVisionV51", "disney"]
}

MODEL_CKPTS = {
    f"hansyan/perflow-sd15-{m}": f"pretrained/perflow-sd15-{m}"
    for m in ["dreamshaper", "realisticVisionV51", "disney"]
}


def download_weights(url, dest, extract=True):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    args = ["pget"]
    if extract:
        args.append("-x")
    subprocess.check_call(args + [url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        for model in MODELS:
            if not os.path.exists(MODEL_URLS[model]):
                download_weights(MODEL_URLS[model], MODEL_CKPTS[model])
        self.pipelines = {
            model: StableDiffusionPipeline.from_pretrained(
                MODEL_CKPTS[model], torch_dtype=torch.float16
            )
            for model in MODELS
        }
        for pipe in self.pipelines.values():
            pipe.scheduler = PeRFlowScheduler.from_config(
                pipe.scheduler.config, prediction_type="epsilon", num_time_windows=4
            )
            pipe.to("cuda", torch.float16)

    @torch.inference_mode()
    def predict(
        self,
        model: str = Input(
            description="Choose a model",
            default="hansyan/perflow-sd15-dreamshaper",
            choices=MODELS,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="A man with brown skin, a beard, and dark eyes",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", default=8
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        pipe = self.pipelines[model]

        generator = torch.Generator("cuda").manual_seed(seed)
        samples = pipe(
            prompt=[prompt],
            negative_prompt=[negative_prompt],
            height=512,
            width=512,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images
        out_path = "/tmp/out.png"
        samples[0].save(out_path)
        return Path(out_path)
