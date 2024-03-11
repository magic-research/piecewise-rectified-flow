# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
import time
from PIL import Image
from cog import BasePredictor, Input, Path
import torch
from diffusers import ControlNetModel
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
CONTROLNET_URL = "https://weights.replicate.delivery/default/piecewise-rectified-flow/control_v11f1e_sd15_tile.tar"
CONTROLNET_CKPT = "pretrained/control_v11f1e_sd15_tile"


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
            if not os.path.exists(MODEL_CKPTS[model]):
                download_weights(MODEL_URLS[model], MODEL_CKPTS[model])
        if not os.path.exists(CONTROLNET_CKPT):
            download_weights(CONTROLNET_URL, CONTROLNET_CKPT)
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
            pipe.to("cuda")
        self.controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_CKPT, torch_dtype=torch.float16
        )

        self.upscalers = {
            model: StableDiffusionPipeline.from_pretrained(
                MODEL_CKPTS[model],
                torch_dtype=torch.float16,
                custom_pipeline="stable_diffusion_controlnet_img2img",
                controlnet=self.controlnet,
            )
            for model in MODELS
        }
        for upscaler in self.upscalers.values():
            upscaler.scheduler = PeRFlowScheduler.from_config(
                upscaler.scheduler.config, prediction_type="epsilon", num_time_windows=4
            )
            upscaler.to("cuda")

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
            default="A colorful bird standing on the tree, open beak",
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
        use_refiner: bool = Input(
            description="Combine with ControlNet-Tile to upscale to 1024x1024",
            default=False,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        pipe = self.pipelines[model]
        upscaler = self.upscalers[model] if use_refiner else None

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
        out_path = "/tmp/output.png"
        if not use_refiner:
            samples[0].save(out_path)
        else:
            condition_image = resize_for_condition_image(samples[0], 1024)
            refined_samples = upscaler(
                prompt="best quality",
                negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality",
                image=condition_image,
                controlnet_conditioning_image=condition_image,
                width=condition_image.size[0],
                height=condition_image.size[1],
                strength=1.0,
                generator=torch.manual_seed(seed),
                num_inference_steps=4,
            ).images
            refined_samples[0].save(out_path)
        return Path(out_path)


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img
