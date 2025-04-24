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
import subprocess
import tqdm
from multiprocessing import Pool

paths = []


def gather_paths(input_dir, output_dir):
    for video in sorted(os.listdir(input_dir)):
        if video.endswith(".mp4"):
            video_basename = video[:-4]
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, f"{video_basename}_%03d.mp4")
            if os.path.isfile(video_output):
                continue
            paths.append([video_input, video_output])
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


def segment_video(video_input, video_output):
    os.makedirs(os.path.dirname(video_output), exist_ok=True)
    command = f"ffmpeg -loglevel error -y -i {video_input} -map 0 -c:v copy -segment_time 5 -f segment -reset_timestamps 1 -q:a 0 {video_output}"
    # command = f'ffmpeg -loglevel error -y -i {video_input} -map 0 -segment_time 5 -f segment -reset_timestamps 1 -force_key_frames "expr:gte(t,n_forced*5)" -crf 18 -q:a 0 {video_output}'
    subprocess.run(command, shell=True)


def multi_run_wrapper(args):
    return segment_video(*args)


def segment_videos_multiprocessing(input_dir, output_dir, num_workers):
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_paths(input_dir, output_dir)

    print(f"Segmenting videos of {input_dir} ...")
    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(multi_run_wrapper, paths), total=len(paths)):
            pass


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/shot"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/segmented"
    num_workers = 50

    segment_videos_multiprocessing(input_dir, output_dir, num_workers)
