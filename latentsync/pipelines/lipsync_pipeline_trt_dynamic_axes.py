# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed, write_video_from_imgs
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

import gc
import contextlib
import wave
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
trt.init_libnvinfer_plugins(None, "")

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipelineTRT(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()
        # vae_engine_path='./trt_engines/vae.trt'
        # self.vae_engine = self.load_trt_engine(vae_engine_path)
        unet_engine_path='./trt_engines/denoising_unet_10_2.trt'
        self.engine = self.load_trt_engine(unet_engine_path)
        self.context = self.engine.create_execution_context()

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            scheduler=scheduler
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")
    
    def load_trt_engine(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def allocate_buffers(self, dynamic_input_shapes):
        bindings = []
        inputs = []
        outputs = []
        
        # # TensorRT 8.5+ 获取绑定数量的方法
        num_bindings = self.engine.num_io_tensors  

        
        for i in range(num_bindings):
            # 获取张量名称
            tensor_name = self.engine.get_tensor_name(i)
            
            # 获取张量信息
            dtype = self.engine.get_tensor_dtype(tensor_name)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                if tensor_name in dynamic_input_shapes:
                    shape = dynamic_input_shapes[tensor_name]
                    self.context.set_input_shape(tensor_name, shape) 
            shape = self.context.get_tensor_shape(tensor_name)
            
            # 分配内存
            size = trt.volume(shape) * np.dtype(trt.nptype(dtype)).itemsize
            device_mem = cuda.mem_alloc(size)
            bindings.append(int(device_mem))
            
            # 判断输入输出
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(device_mem)
            else:
                outputs.append(device_mem)
        
        return bindings, inputs, outputs

    def denoising_unet_trt_infer(self, denoising_unet_input, t, audio_embeds):
        
        denoising_unet_input = denoising_unet_input.cpu().numpy().astype(np.float16)
        t = t.cpu().numpy().astype(np.int32)
        audio_embeds = audio_embeds.cpu().numpy().astype(np.float16)
        dynamic_input_shapes = {
            "audio_embeds": audio_embeds.shape
        }
        # 分配缓冲区
        bindings, inputs, outputs = self.allocate_buffers(dynamic_input_shapes)

        # 将数据从主机复制到 GPU
        cuda.memcpy_htod(inputs[0], denoising_unet_input)  
        cuda.memcpy_htod(inputs[1], t)  
        cuda.memcpy_htod(inputs[2], audio_embeds)

        # 执行推理
        self.context.execute_v2(bindings)
        
        # 获取输出并将其从 GPU 拷贝到主机
        noise_pred = np.empty((2, 4, 16, 32, 32), dtype=np.float16) # torch.Size([2, 4, 8, 32, 32])
        cuda.memcpy_dtoh(noise_pred, outputs[0]) # 2 4 16 32 32
        noise_pred_cuda = torch.tensor(noise_pred, dtype=torch.float16, device='cuda')
        return noise_pred_cuda
     
    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.denoising_unet, "_hf_hook"):
            return self.device
        for module in self.denoising_unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list, start_idx: int = 0, LOOP_COEFF: int = 0):
        end_idx = start_idx + LOOP_COEFF * len(faces)
        video_frames = video_frames[start_idx: end_idx]
        boxes = boxes[start_idx: end_idx]
        affine_matrices = affine_matrices[start_idx: end_idx]
        # video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces, position=2)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()
            # face = cv2.resize(face, (width, height), interpolation=cv2.INTER_LANCZOS4)
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)
    
    def restore_video2imgs(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list, video_frames_dir: str):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()
            # face = cv2.resize(face, (width, height), interpolation=cv2.INTER_LANCZOS4)
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        
        print('Write frames into folder')
        output_dir = video_frames_dir
        os.makedirs(output_dir, exist_ok=True)
        for index, out_frame in enumerate(tqdm.tqdm(out_frames)):
            frame_filename = os.path.join(output_dir, f"frame_{index:05d}.png")
            cv2.imwrite(frame_filename, out_frame)
        return
    
    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices

    def get_wav_duration(self, wav_path):
        with contextlib.closing(wave.open(wav_path, 'r')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
        return duration
    
    @torch.no_grad()
    def __call__(
        self,
        args,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        mask_image_path: str = "latentsync/utils/mask.png",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):

        check_ffmpeg_installed()

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.denoising_unet.config.sample_size * self.vae_scale_factor
        width = width or self.denoising_unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # whisper_chunks 18min音频特征 27976帧 占用 1G内存
        duration = self.get_wav_duration(audio_path)
        ONE_HOUR = 3600
        if duration > ONE_HOUR:
            raise ValueError(f"audio time = {duration:.2f}s > 1 hour, out of limits")
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
        
        # workspace dir
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_video_index = 0
        temp_videos = []

        # 每批处理 x 音帧
        AUDIO_FRAMES_BATCH = min(len(whisper_chunks), 500)
        video_load_frames = min(len(whisper_chunks), 1000)
        video_frames = read_video(video_path, use_decord=False, max_frames=video_load_frames)
        video_frames = video_frames[::-1]
        video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks[0: AUDIO_FRAMES_BATCH], video_frames)
        
        for start_idx in tqdm.tqdm(range(0, len(whisper_chunks), AUDIO_FRAMES_BATCH), position=0, desc="Processing audio batches..."):
            
            end_idx = min(start_idx + AUDIO_FRAMES_BATCH, len(whisper_chunks))
            current_chunks = whisper_chunks[start_idx: end_idx]

            # For  video_frames: the TAIL of first chunks should be consist with the HEAD of the next chunks
            # Loop 0...499 499...0 0...499 499...0
            video_frames = video_frames[::-1][:len(current_chunks)]
            faces = torch.flip(faces, [0])[:len(current_chunks)]
            boxes = boxes[::-1][:len(current_chunks)]
            affine_matrices = affine_matrices[::-1][:len(current_chunks)]
            num_channels_latents = self.vae.config.latent_channels

            # Prepare latent variables, 1*4*len(current_chunks)*32*32
            all_latents = self.prepare_latents(
                batch_size,
                len(current_chunks),
                num_channels_latents,
                height,
                width,
                weight_dtype,
                device,
                generator,
            )

            num_inferences = math.ceil(len(current_chunks) / num_frames)
            LOOP_COEFF = 4
            synced_video_frames = []
            cnt = 0
            for i in tqdm.tqdm(range(num_inferences), position=1, desc="Doing inference..."):

                audio_embeds = torch.stack(current_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])

                inference_faces = faces[i * num_frames : (i + 1) * num_frames]
                latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
                ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                    inference_faces, affine_transform=False
                )

                # 7. Prepare mask latent variables
                mask_latents, masked_image_latents = self.prepare_mask_latents(
                    masks,
                    masked_pixel_values,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                )

                # 8. Prepare image latents
                ref_latents = self.prepare_image_latents(
                    ref_pixel_values,
                    device,
                    weight_dtype,
                    generator,
                    do_classifier_free_guidance,
                )

                # 9. Denoising loop
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for j, t in enumerate(timesteps):
                        # expand the latents if we are doing classifier free guidance
                        denoising_unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                        denoising_unet_input = self.scheduler.scale_model_input(denoising_unet_input, t)

                        # concat latents, mask, masked_image_latents in the channel dimension
                        denoising_unet_input = torch.cat(
                            [denoising_unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                        )

                        # predict the noise residual
                        noise_pred = self.denoising_unet_trt_infer(denoising_unet_input, t, audio_embeds)
                        # noise_pred = self.denoising_unet(
                        #     denoising_unet_input, t, encoder_hidden_states=audio_embeds
                        # ).sample

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                        # call the callback, if provided
                        if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and j % callback_steps == 0:
                                callback(j, t, latents)

                # Recover the pixel values
                decoded_latents = self.decode_latents(latents)
                decoded_latents = self.paste_surrounding_pixels_back(
                    decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
                )
                synced_video_frames.append(decoded_latents)
                
                if (i + 1) % LOOP_COEFF == 0:
                    synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices, num_frames*cnt*LOOP_COEFF, LOOP_COEFF)

                    temp_video_path = os.path.join(temp_dir, f"temp_{temp_video_index}.mp4")
                    temp_videos.append(f"temp_{temp_video_index}.mp4")
                    write_video(temp_video_path, synced_video_frames, fps=25)
                    temp_video_index = temp_video_index + 1
                    synced_video_frames = []
                    cnt = cnt + 1

            # 处理剩余的未保存帧
            if synced_video_frames:
                synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices, num_frames * cnt * LOOP_COEFF, LOOP_COEFF)

                temp_video_path = os.path.join(temp_dir, f"temp_{temp_video_index}.mp4")
                temp_videos.append(f"temp_{temp_video_index}.mp4")
                write_video(temp_video_path, synced_video_frames, fps=25)
                temp_video_index += 1
                synced_video_frames = []
        
        if len(temp_videos) > 1:
            concat_file = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_file, "w") as f:
                for video in temp_videos:
                    f.write(f"file '{video}'\n")

            concat_video_out = os.path.join(temp_dir, 'temp_all.mp4')
            command = f"ffmpeg -y -f concat -safe 0 -i {concat_file} -c copy {concat_video_out}"
            subprocess.run(command, shell=True)
            print(f"Final video saved: {concat_video_out}")
        
        audio_samples = read_audio(audio_path)
        audio_samples_remain_length = int(len(whisper_chunks) / video_fps * audio_sample_rate) # len(video_frames)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()
        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)
        print('Start to synthesize video with audio')
        command = f"ffmpeg -y -nostdin -i {concat_video_out} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
