# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from datetime import datetime
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import argparse
import os
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.pipelines.lipsync_pipeline_trt import LipsyncPipelineTRT
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature

def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")

    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Output video path: {args.video_out_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained("./checkpoints/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    # denoising_unet, _ = UNet3DConditionModel.from_pretrained(
    #     OmegaConf.to_container(config.model),
    #     args.inference_ckpt_path,
    #     device="cpu",
    # )

    # denoising_unet = denoising_unet.to(dtype=dtype)

    pipeline = LipsyncPipelineTRT(
        vae=vae,
        audio_encoder=audio_encoder,
        scheduler=scheduler
    ).to("cuda")

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    pipeline(
        args = args,
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
    )


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/stage2.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, default='./checkpoints/latentsync_unet.pt')
    parser.add_argument("--video_path", type=str, default='./data/wwj_3s.mp4')
    parser.add_argument("--audio_path", type=str, default='./data/intro_6s.wav')
    parser.add_argument("--video_out_path", type=str, default='')
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1247)
    args = parser.parse_args()

    if not args.video_out_path:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]
        args.video_out_path = f"./data/ai-{video_name}-{audio_name}.mp4"
    config = OmegaConf.load(args.unet_config_path)
    main(config, args)

    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f'Finish Processing at cost of time {elapsed_time}s')