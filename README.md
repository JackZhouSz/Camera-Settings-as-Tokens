# Camera Settings as Tokens
[[paper]](https://dl.acm.org/doi/10.1145/3680528.3687635)[[project page]](https://camera-settings-as-tokens.github.io/)[[demo]](https://huggingface.co/spaces/Camera-Settings-as-Tokens/Camera-Settings-as-Tokens)[[dataset]](https://github.com/aiiu-lab/CameraSettings20K)[[model]](https://huggingface.co/ishengfang/Camera-Settings-as-Tokens-SD2)

Offical code for our SIGGRAPH Asia 2024 paper, Camera Settings as Tokens: Modeling Photography on Latent Diffusion Models

## TL;DR: Camera Settings 📷 + Text 📝 ⮕ Image 🖼️ 

![](https://camera-settings-as-tokens.github.io/static/images/teaser.png)


## Requirements
We highly recommend using the [Conda](https://docs.anaconda.com/miniconda/) to build the environment. 

You can build and activate the environment by following commands. 
```bash
conda env create -f env.yml 
conda activate Camera-Settings-as-Tokens
```

## Text-n-Camera-Settngs-to-Image Generation
We provide the code for text-to-image generation with the pre-trained model. 

### Usage
```bash
python text+cam2image.py --prompt "half body portrait of a beautiful Portuguese woman, pale skin, brown hair with blonde highlights, wearing jeans, nature and cherry blossom trees in background" \
--negative_prompt "ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra legs, mutated hands fused fingers, too many fingers, long neck" \
--focal_length 50 --f_number 1.8 --iso_speed_rating 100 --exposure_time 0.01 --output_basename "woman_cherry_blossom_trees" --lora_scale 1.0
```
For imaginary prompt, such as "astronaut riding a horse on the moon", we recommand to set the `--lora_scale` to 0.5.

### Results
![](./results/woman_cherry_blossom_trees+50mm_f4_0_ISO100_ET0_01_seed87.png)

## Text-n-Camera-Settngs-to-Image Generation with ControlNet
We provide the code for text-to-image generation with ControlNet.
Notice that due to differences in the implementation of LoRA merging, the recommended LoRA scale is 0.55 instead of 0.275 as reported in the appendix of our SIGGRAPH Asia 2024 paper.

### Usage
#### With Image for Conditioning
```bash
python text+cam2image_w_controlnet.py --prompt "a cute cat, nature and cherry blossom trees in background" \
--image_for_conditioning <path for image for conditioning> \
--negative_prompt "ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra legs, mutated hands fused fingers, too many fingers, long neck" \
--focal_length 50 --f_number 1.8 --iso_speed_rating 100 --exposure_time 0.01 --output_basename "cat_cherry_blossom_trees" --lora_scale 0.55
```

#### With Conditional Image
```bash
python text+cam2image_w_controlnet.py --prompt "a cute cat, nature and cherry blossom trees in background" \
--conditional_image results/depth_cat_cherry_blossom_trees.png \
--negative_prompt "ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra legs, mutated hands fused fingers, too many fingers, long neck" \
--focal_length 50 --f_number 1.8 --iso_speed_rating 100 --exposure_time 0.01 --output_basename "cat_cherry_blossom_trees" --lora_scale 0.55
```

### Results
![](./results/depth_cat_cherry_blossom_trees.png)
![](./results/cat_cherry_blossom_trees+50mm_f1_8_ISO100_ET0_01_seed87.png)

## Training

We provide the code for training the model. 

### Usage
```bash
accelerate launch train_cam+text2image_lora.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1" \
  --dataset_name <dataset_path>\
  --caption_column "text" \
  --resolution 512 --random_flip \
  --train_batch_size <batch_size> --gradient_accumulation_steps <gradient_accumulation_steps> \
  --num_train_epochs 100 --checkpointing_steps 500 \
  --learning_rate 1e-04 --lr_scheduler "constant" --lr_warmup_steps 0 \
  --dataloader_num_workers <number of workers> \
  --output_dir <output_path for logs and weights> \
  --validation_prompt <validation_prompt (seperate with space)> \
  --validation_focal_length <validation_focal_length (seperate with space)> \
  --validation_f_number <validation_f_number (seperate with space)> \
  --validation_iso_speed_rating <validation_iso_speed_rating (seperate with space)> \
  --validation_exposure_time <validation_exposure_time (seperate with space)> \
  --cam_embed_embedding_dim 1024 \
  --valid_seed 87
```

We recommend to set the total batch size to 128. For Stable Diffusion 2, please set `cam_embed_embedding_dim` to 1024 to fit the output dimension of the OpenCLIP text encoder.  For Stable Diffusion 1, please set `cam_embed_embedding_dim` to 768 to fit the output dimension of the CLIP text encoder.

## ToDo
- [ ] Code for image editing
- [x] Code for inference with ControlNet
- [x] Add the code for the training
- [ ] SDXL

## Notice
This code and model are for research only. For other purposes, please contact us.

## Citation
```Bibtex
@inproceedings{fang2024camera,
  title={Camera Settings as Tokens: Modeling Photography on Latent Diffusion Models},
  author={I-Sheng Fang and Yue-Hua Han and Jun-Cheng Chen},
  booktitle={SIGGRAPH Asia},
  year={2024}
}
```
