import random, argparse, os
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torchvision
from diffusers import UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline
from diffusers.utils import make_image_grid, load_image

from src.utils_perflow import merge_delta_weights_into_unet
from src.scheduler_perflow import PeRFlowScheduler

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
PROMPT_LIST_DEPTH = [["assets/others/control/depth/husky.jpg", ["A husky dog, lying on his stomach in the snow, looking away, blue sky","A cute cat, lying on the ground, raining outside"]],]
PROMPT_LIST_TILE = [
    ["assets/others/control/dog.png", ["a dog sitting",]],
    ["assets/others/control/fruits.png", ["a plate of fruits",]],
]

def main(args):
    save_dir = os.path.join(args.save_dir + f"_{args.num_inference_steps}")
    Path(os.path.join(save_dir, "images")).mkdir(parents=True, exist_ok=True)

    ## controlnet pipeline
    from scripts.controlnet_preprocessor import Preprocessor
    controlnet_preprocessor = Preprocessor(control_type=args.control_type, path="lllyasviel/Annotators")
    if args.control_type == "openpose":
        controlnet_model_path = "lllyasviel/control_v11p_sd15_openpose"
        controlnet_preprocessor.preprocessor.to(device='cuda')
    elif args.control_type == "midas":
        controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
        controlnet_preprocessor.preprocessor.to(device='cuda')
    elif args.control_type == "canny":
        controlnet_model_path = "lllyasviel/control_v11p_sd15_canny"
    elif args.control_type == 'tile':
        controlnet_model_path = "lllyasviel/control_v11f1e_sd15_tile"
    else:
        raise NotImplementedError

    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
    if args.control_type == 'tile':
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(args.sd_base, controlnet=controlnet, torch_dtype=torch.float16,)
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(args.sd_base, controlnet = controlnet, torch_dtype=torch.float16,)
        
    delta_weights = UNet2DConditionModel.from_pretrained("hansyan/piecewise-rectified-flow-delta-weights", torch_dtype=torch.float16, variant="v0-1",).state_dict()
    pipe = merge_delta_weights_into_unet(pipe, delta_weights)
    pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="epsilon", num_time_windows=4,)
    
    pipe.to("cuda", torch.float16)


    ## sampling
    if args.control_type == "midas":
        prompts_list = PROMPT_LIST_DEPTH
    elif args.control_type == 'tile':
        prompts_list = PROMPT_LIST_TILE
    else:
        raise NotImplementedError

    prompt_prefix = "raw photo, 8k uhd, dslr, high quality, hyper detailed masterpiece; "
    negative_prompt = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast"

    assert args.guidance_scale is not None
    if "-" in args.guidance_scale:
        cfg_scale_list = [float(x) for x in args.guidance_scale.split("-")]
    else:
        cfg_scale_list = [float(args.guidance_scale)]

    for cfg_scale in cfg_scale_list:
        for i, prompt_ctrl in enumerate(prompts_list):
            setup_seed(args.seed)
            ctrl, prompts = prompt_ctrl[0], prompt_ctrl[1]
            prompts = [prompt_prefix + p for p in prompts]
            
            if args.control_type == 'tile':
                ctrl = load_image(ctrl)
                print(f"low res input: {ctrl.size}, upsampling to ---> {args.size}")
                ctrl = controlnet_preprocessor(np.array(ctrl), image_resolution=args.size)
                ctrl = Image.fromarray(ctrl)
                samples = pipe(
                            image               = ctrl,
                            control_image       = ctrl, 
                            strength            = 1.0,
                            prompt              = prompts, 
                            negative_prompt     = [negative_prompt] * len(prompts),
                            height              = args.size,
                            width               = args.size,
                            num_inference_steps = args.num_inference_steps, 
                            guidance_scale      = cfg_scale,
                            output_type         = 'pt',
                            ).images
            else:
                ctrl = load_image(ctrl)
                ctrl = controlnet_preprocessor(np.array(ctrl), image_resolution=args.size)
                ctrl = Image.fromarray(ctrl)
                samples = pipe(
                    image               = ctrl,
                    prompt              = prompts, 
                    negative_prompt     = [negative_prompt] * len(prompts),
                    height              = args.size,
                    width               = args.size,
                    num_inference_steps = args.num_inference_steps, 
                    guidance_scale      = cfg_scale,
                    output_type         = 'pt',
                ).images
            
            cfg_int = int(cfg_scale); cfg_float = int(cfg_scale*10 - cfg_int*10)
            save_name = f'ctrl{i+1}_cfg{cfg_int}-{cfg_float}.png'
            torchvision.utils.save_image(
                torchvision.utils.make_grid(samples, nrow = 4), 
                os.path.join(save_dir, save_name)
            )
            ctrl.save(os.path.join(save_dir, f'ctrl{i+1}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default='demo',)
    parser.add_argument("--control_type", type=str, default='tile',)
    parser.add_argument("--sd_base", type=str, default="Lykon/dreamshaper-8",)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=4, help="number of inference steps")
    parser.add_argument("--guidance_scale", type=str, default="4.5", help="number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="number of inference steps")
    args = parser.parse_args()
    main(args)