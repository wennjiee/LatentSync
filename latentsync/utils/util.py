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

import os
import numpy as np
import json
from typing import Union
from pathlib import Path
import matplotlib.pyplot as plt
import imageio

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
from torchvision import transforms

from einops import rearrange
import cv2
from decord import AudioReader, VideoReader
import shutil
import subprocess
import tqdm
import math
from glob import glob

# Machine epsilon for a float32 (single precision)
eps = np.finfo(np.float32).eps


def read_json(filepath: str):
    with open(filepath) as f:
        json_dict = json.load(f)
    return json_dict


def get_video_resolution(video_path):
    command = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    video_info = json.loads(result.stdout)
    width = video_info["streams"][0]["width"]
    height = video_info["streams"][0]["height"]
    return width, height


def read_video(video_path: str, change_fps=True, use_decord=True, max_frames=-1):
    if change_fps:
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        width, height = get_video_resolution(video_path)
        print(f'width={width}, height={height}')
        scale_option = ""
        # check if resolution > 1080P, convert it to 1080P
        # if height > 1080:
        #     scale_option = "-vf scale=-1:1080"
        # 考虑显卡加速
        command = (
            f"ffmpeg -y -nostdin -i {video_path} {scale_option} -r 25 -crf 18 -threads 8 {os.path.join(temp_dir, 'video.mp4')}"
        )
        print(f'cmd = {command}')
        subprocess.run(command, shell=True)
        target_video_path = os.path.join(temp_dir, "video.mp4")
    else:
        target_video_path = video_path
    print(f'Start reading video with method use_decord:{use_decord}')
    if use_decord:
        return read_video_decord(target_video_path, max_frames)
    else:
        return read_video_cv2(target_video_path, max_frames)


def split_video_and_audio(workspace: str, video_path: str, audio_path: str, segment_frames: int, fps: int):
    """
    Preprocessing function for video and audio. 
    It segments the media to enable execution under low-resource conditions.
    """
    # 1. Video splitting
    width, height = get_video_resolution(video_path)
    print(f'width={width}, height={height}')
    scale_option = []
    # Resizing the resolution may result in aspect ratio distortion.
    # check if resolution > 1080P, convert it to 1080P
    # if height > 1080:
    #     scale_option = "-vf scale=-1:1080"
    # Maybe accelerate using nvml
    segments_path = os.path.join(workspace, "segments")
    os.makedirs(segments_path, exist_ok=True)
    segment_seconds = int(segment_frames / fps)
    command = [
        "ffmpeg", "-y",
        "-nostdin", "-i", video_path,
        *scale_option, "-r", "25",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-threads", "8",
        "-force_key_frames", f"expr:gte(t,n_forced*{segment_seconds})",
        "-c:a", "copy",
        "-f", "segment",
        "-segment_time", f"{segment_seconds}",
        "-reset_timestamps", "1",
        os.path.join(segments_path, "chunk_%03d.mp4"),
    ]
    print(f'cmd = {command}')
    subprocess.run(command, shell=False)
    video_files = sorted(glob(os.path.abspath(os.path.join(segments_path, "chunk_*.mp4"))))
    
    # 2. Audio splitting
    res_video_files = []
    res_audio_files = []
    print(f"[INFO] Extracting audio from {len(video_files)} segments...")
    
    for video_file in video_files:
        if has_audio(video_path=video_file):
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            wav_file = os.path.abspath(os.path.join(segments_path, f"{base_name}.wav"))
            
            extract_cmd = [
                "ffmpeg", "-y", "-i", video_file,
                "-vn", "-loglevel", "warning",
                "-threads", "8",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "2",
                wav_file
            ]
            subprocess.run(extract_cmd, check=True)
            res_video_files.append(video_file)
            res_audio_files.append(wav_file)
    print(f"[INFO] Finish Extracting Audio")
    
    return res_video_files, res_audio_files


def loop_video_to_match_audio(workspace: str, input_video: str, input_audio: str):
    """
    Loops the video in alternating forward and reverse order until it exceeds the audio length, 
    Trims it to match the audio duration.
    """
    os.makedirs(workspace, exist_ok=True)

    def get_media_duration(path):
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return float(result.stdout)

    def create_reversed_video(src, dst):
        cmd = [
            'ffmpeg', '-y', '-i', src,
            '-vf', 'reverse',
            '-an', "-threads", "8",
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            dst
        ]
        subprocess.run(cmd, check=True)

    def create_forward_video(src, dst):
        cmd = [
            'ffmpeg', '-y', '-i', src,
            '-an', "-threads", "8",
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            dst
        ]
        subprocess.run(cmd, check=True)

    def create_concat_file(n, fwd_file, rev_file, list_path):
        with open(list_path, 'w') as f:
            for i in range(n):
                file_path = fwd_file if i % 2 == 0 else rev_file
                f.write(f"file '{os.path.abspath(file_path)}'\n")
    
    print("Getting video and audio durations...")
    video_duration = get_media_duration(input_video)
    audio_duration = get_media_duration(input_audio)
    print(f"Video duration: {video_duration}s, Audio duration: {audio_duration}s")


    # Calculate the number of loops to ensure the video duration >= audio duration
    n_loops = math.ceil(audio_duration / video_duration)
    forward_video = os.path.join(workspace, "forward.mp4")
    reverse_video = os.path.join(workspace, "reverse.mp4")
    concat_list = os.path.join(workspace, "loop_concat_list.txt")
    loop_video = os.path.join(workspace, "loop_video.mp4")

    create_reversed_video(input_video, reverse_video)
    create_forward_video(input_video, forward_video)
    create_concat_file(n_loops, forward_video, reverse_video, concat_list)

    subprocess.run([
        'ffmpeg', '-y', "-threads", "8", '-f', 'concat', '-safe', '0', '-i', concat_list,
        '-c', 'copy', loop_video
    ], check=True)

    standard_video = os.path.join(workspace, 'input.mp4')
    subprocess.run([
        'ffmpeg', '-y',
        '-i', loop_video,
        '-i', input_audio,
        "-threads", "8",
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'aac',
        '-shortest',
        standard_video
    ], check=True)
    
    print(f"Merging completed, File: {standard_video}")
    return standard_video, input_audio


def has_audio(video_path: str) -> bool:
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a',
        '-show_entries',
        'stream=codec_type',
        '-of', 'csv=p=0',
        video_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
    return bool(res.stdout.strip())


def read_video_decord(video_path: str, max_frames: int):
    vr = VideoReader(video_path)
    print('Reading video...')
    try:
        video_frames = vr[:max_frames].asnumpy()
        vr.seek(0)
    except Exception as e:
        print(f'Exception ocurred, E is {e}')
        video_frames = []
        return video_frames
    print('Finish reading video...')
    return video_frames


def read_video_cv2(video_path: str, max_frames: int):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return np.array([])

    frames = []

    try:
        while True:
            # Read a frame
            ret, frame = cap.read()

            # If frame is read correctly ret is True
            if not ret or len(frames) > max_frames:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame_rgb)
    except Exception as e:
        print(f'Exception ocurred, E is {e}')
        return np.array([])
    finally:
        # Release the video capture object
        cap.release()
    return np.array(frames)


def read_audio(audio_path: str, audio_sample_rate: int = 16000):
    if audio_path is None:
        raise ValueError("Audio path is required.")
    ar = AudioReader(audio_path, sample_rate=audio_sample_rate, mono=True)

    # To access the audio samples
    audio_samples = torch.from_numpy(ar[:].asnumpy())
    audio_samples = audio_samples.squeeze(0)

    return audio_samples


def write_video(video_output_path: str, video_frames: np.ndarray, fps: int):
    with imageio.get_writer(
        video_output_path,
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", "13"],
        ffmpeg_log_level="error",
    ) as writer:
        for video_frame in video_frames:
            writer.append_data(video_frame)


def write_video_cv2(video_output_path: str, video_frames: np.ndarray, fps: int):
    height, width = video_frames[0].shape[:2]
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    # out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"vp09"), fps, (width, height))
    for frame in video_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def init_dist(backend="nccl", **kwargs):
    """Initializes distributed environment."""
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for training.")
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)

    return local_rank


def zero_rank_print(s):
    if dist.is_initialized() and dist.get_rank() == 0:
        print("### " + s)


def zero_rank_log(logger, message: str):
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info(message)


def check_video_fps(video_path: str):
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        raise ValueError(f"Video FPS is not 25, it is {fps}. Please convert the video to 25 FPS.")


def one_step_sampling(ddim_scheduler, pred_noise, timesteps, x_t):
    # Compute alphas, betas
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timesteps].to(dtype=pred_noise.dtype)
    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/abs/2010.02502
    if ddim_scheduler.config.prediction_type == "epsilon":
        beta_prod_t = beta_prod_t[:, None, None, None, None]
        alpha_prod_t = alpha_prod_t[:, None, None, None, None]
        pred_original_sample = (x_t - beta_prod_t ** (0.5) * pred_noise) / alpha_prod_t ** (0.5)
    else:
        raise NotImplementedError("This prediction type is not implemented yet")

    # Clip "predicted x_0"
    if ddim_scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
    return pred_original_sample


def plot_loss_chart(save_path: str, *args):
    # Creating the plot
    plt.figure()
    for loss_line in args:
        plt.plot(loss_line[1], loss_line[2], label=loss_line[0])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()

    # Save the figure to a file
    plt.savefig(save_path)

    # Close the figure to free memory
    plt.close()


CRED = "\033[91m"
CEND = "\033[0m"


def red_text(text: str):
    return f"{CRED}{text}{CEND}"


log_loss = nn.BCELoss(reduction="none")


def cosine_loss(vision_embeds, audio_embeds, y):
    sims = nn.functional.cosine_similarity(vision_embeds, audio_embeds)
    # sims[sims!=sims] = 0 # remove nan
    # sims = sims.clamp(0, 1)
    loss = log_loss(sims.unsqueeze(1), y).squeeze()
    return loss


def save_image(image, save_path):
    # input size (C, H, W)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).to(torch.uint8)
    image = transforms.ToPILImage()(image)
    # Save the image copy
    image.save(save_path)

    # Close the image file
    image.close()


def gather_loss(loss, device):
    # Sum the local loss across all processes
    local_loss = loss.item()
    global_loss = torch.tensor(local_loss, dtype=torch.float32).to(device)
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)

    # Calculate the average loss across all processes
    global_average_loss = global_loss.item() / dist.get_world_size()
    return global_average_loss


def gather_video_paths_recursively(input_dir):
    print(f"Recursively gathering video paths of {input_dir} ...")
    paths = []
    gather_video_paths(input_dir, paths)
    return paths


def gather_video_paths(input_dir, paths):
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".mp4"):
            filepath = os.path.join(input_dir, file)
            paths.append(filepath)
        elif os.path.isdir(os.path.join(input_dir, file)):
            gather_video_paths(os.path.join(input_dir, file), paths)


def count_video_time(video_path):
    video = cv2.VideoCapture(video_path)

    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    return frame_count / fps


def check_ffmpeg_installed():
    # Run the ffmpeg command with the -version argument to check if it's installed
    result = subprocess.run("ffmpeg -version", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if not result.returncode == 0:
        raise FileNotFoundError("ffmpeg not found, please install it by:\n    $ conda install -c conda-forge ffmpeg")


def check_model_and_download(ckpt_path: str, huggingface_model_id: str = "ByteDance/LatentSync-1.5"):
    if not os.path.exists(ckpt_path):
        ckpt_path_obj = Path(ckpt_path)
        download_cmd = f"huggingface-cli download {huggingface_model_id} {Path(*ckpt_path_obj.parts[1:])} --local-dir {Path(ckpt_path_obj.parts[0])}"
        subprocess.run(download_cmd, shell=True)


class dummy_context:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
