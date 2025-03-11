import argparse
import logging
import math
import os

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from camera_embed import CameraSettingEmbedding
from inference import embed_camera_settings
from diffusers.schedulers import PNDMScheduler
import random

import torch
from safetensors import safe_open
import os
from PIL import Image



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2" , help="Stable Diffusion Model ID")
    parser.add_argument("--camera_setting_embedding_id", type=str, default="ishengfang/Camera-Settings-as-Tokens-SD2" , 
                        help="Camera Setting Embedding Model ID")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--output_basename", type=str, default="cat_cherry_blossom_trees", help="Output file basename")

    parser.add_argument("--contoronet_model_id", type=str, default="thibaud/controlnet-sd21-depth-diffusers" , 
                        help="ControlNet Model ID", )
    parser.add_argument("--image_for_conditioning", type=str, default=None, help="Image for conditioning")
    parser.add_argument("--conditional_image", type=str, default=None, help="Conditional image for ControlNet")

    parser.add_argument("--lora_scale", type=float, default=0.55, )

    parser.add_argument("--prompt", type=str, help="Prompt for the model",
                        default="a cute cat, nature and cherry blossom trees in background")
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt for the model",
                        default="ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra legs, mutated hands fused fingers, too many fingers, long neck")

    parser.add_argument("--seed", type=int, default=87, help="Random seed")

    parser.add_argument("--focal_length", type=float, default=50.0, )
    parser.add_argument("--f_number", type=float, default=4.0, )
    parser.add_argument("--iso_speed_rating", type=float, default=100.0, )
    parser.add_argument("--exposure_time", type=float, default=0.01 )


    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    controlnet = ControlNetModel.from_pretrained(args.contoronet_model_id)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(args.model_id, controlnet=controlnet)
    pipeline.scheduler = PNDMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    cam_embed = CameraSettingEmbedding.from_pretrained(args.camera_setting_embedding_id, subfolder="cam_embed")
    pipeline.load_lora_weights(args.camera_setting_embedding_id, adapter_name="camera-settings-as-tokens")
    pipeline.set_adapters(["camera-settings-as-tokens"], adapter_weights=[args.lora_scale])
    pipeline.to(args.device)
    cam_embed.to(args.device)

    if args.image_for_conditioning is not None:
        image_for_conditioning = Image.open(args.image_for_conditioning)
        from controlnet_aux import MidasDetector
        midas_detector = MidasDetector.from_pretrained('lllyasviel/ControlNet')
        conditional_image = midas_detector(image_for_conditioning)
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
        conditional_image.save(os.path.join(args.output_dir, f'depth_{args.output_basename}.png'))
        print(f"Conditional image saved as depth_{args.output_basename}.png")
    if args.conditional_image is not None:
        conditional_image = Image.open(args.conditional_image)

    print(f"Generating image with focal length {args.focal_length}mm and f/{args.f_number}")
    print(f"iso_speed_rating: {args.iso_speed_rating}, exposure_time: {args.exposure_time}")
    print(f"Seed: {args.seed}")
    generator = torch.Generator(device=pipeline._execution_device)
    if args.seed is not None:
        generator.manual_seed(args.seed)
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            args.prompt, args.device, 1, True, negative_prompt=args.negative_prompt)
        prompt_embeds, negative_prompt_embeds = embed_camera_settings(args.focal_length, args.f_number, 
                                                                      args.iso_speed_rating, args.exposure_time, 
                                                                      prompt_embeds=prompt_embeds, 
                                                                      negative_prompt_embeds=negative_prompt_embeds, 
                                                                      cam_embed=cam_embed, device=args.device)
        image = pipeline(image=conditional_image,
                         prompt_embeds=prompt_embeds,
                         negative_prompt_embeds=negative_prompt_embeds, 
                         num_inference_steps=args.num_inference_steps,
                         generator=generator).images[0]
    focal_length = str(int(args.focal_length))
    f_number = str(args.f_number).replace('.','_')
    ISO_speed_rating = str(int(args.iso_speed_rating))
    exposure_time = str(args.exposure_time).replace('.','_')
    save_name = f'{args.output_basename}+{focal_length}mm_f{f_number}_ISO{ISO_speed_rating}_ET{exposure_time}'
    if args.seed is not None:
        save_name += f'_seed{args.seed}'
    save_name += '.png'
    image.save(os.path.join(args.output_dir, save_name))

if __name__ == "__main__":
    main()