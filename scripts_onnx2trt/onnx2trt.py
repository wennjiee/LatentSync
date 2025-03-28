import torch
import onnx
import onnxsim
import sys
from datetime import datetime
import onnx
import onnxsim
import tensorrt as trt
import argparse
import os
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature


def export_denoising_unet_to_onnx(denoising_unet_model, onnx_path):
    denoising_unet_model.to('cuda')
    denoising_unet_model.eval()
    
    # 创建符合 forward() 要求的输入
    dummy_input = torch.randn(2, 13, 16, 32, 32).to('cuda').to(torch.float16)  # 固定形状 torch.Size([2, 13, 8, 32, 32])
    dummy_timestep = torch.tensor(981, dtype=torch.int32, device='cuda')  # 固定形状
    dummy_audio_embeds = torch.randn(32, 50, 384).to('cuda').to(torch.float16)

    # 仅 audio_embeds 允许动态 batch 维度
    dynamic_axes = {
        'audio_embeds': {0: 'audio_batch'}  # 仅对 audio_embeds 的 batch 维度设置动态
    }

    # 导出 ONNX
    torch.onnx.export(
        denoising_unet_model,
        (dummy_input, dummy_timestep, dummy_audio_embeds),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input', 'timestep', 'audio_embeds'],
        output_names=['output'],
        dynamic_axes=dynamic_axes  # 仅 audio_embeds 允许动态 batch
    )

    print(f"Denoising UNet model exported to {onnx_path}")


def build_unet_tensorrt_engine(onnx_file, engine_file):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    try:
        with open(onnx_file, "rb") as f:
            onnx_model = f.read()
            if not parser.parse(model=onnx_model, path='E://_TalkingFaceSys//LatentSync//trt_engines//denoising_unet_onnx_wrapper_16//'):
                print("ERROR: Failed to parse ONNX file")
                for err in range(parser.num_errors):
                    print(parser.get_error(err))
                return None
    except FileNotFoundError:
        print(f"ERROR: File {onnx_file} not found")
        return None
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)  # 启用 FP16（需 GPU 支持）
    profile = builder.create_optimization_profile()

    input_2 = network.get_input(2)  # 你的动态输入

    profile.set_shape(input_2.name, min=(1, 50, 384), opt=(32, 50, 384), max=(32, 50, 384))

    config.add_optimization_profile(profile)

    # 构建 TensorRT 引擎
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return None


    with open(engine_file, "wb") as f:
        f.write(serialized_engine)  # 序列化并保存引擎

    print(f"TensorRT engine for UNet saved to {engine_file}")

def export_vae_to_onnx(vae_model, onnx_path):
    vae_model.to('cuda')
    vae_model.eval()

    # 创建一个与实际输入尺寸一致的假输入
    dummy_input = torch.randn(16, 3, 256, 256).to(torch.float16).to('cuda')
    
    # 导出为 ONNX 格式
    torch.onnx.export(
        vae_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        verbose=True
    )

    model_onnx = onnx.load(onnx_path)
    model_onnx, _ = onnxsim.simplify(model_onnx)
    onnx.save(model_onnx, onnx_path)
    
    print(f"VAE model exported to {onnx_path}")

def build_vae_tensorrt_engine(onnx_file, engine_file):
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX file")
            for err in range(parser.num_errors):
                print(parser.get_error(err))
            return None

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    
    engine = builder.build_engine(network, config)
    with open(engine_file, "wb") as f:
        f.write(engine.serialize())

    print(f"TensorRT engine for VAE saved to {engine_file}")

def main(config, args):

    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    # vae = AutoencoderKL.from_pretrained("./checkpoints/sd-vae-ft-mse", torch_dtype=dtype)
    # vae.config.scaling_factor = 0.18215
    # vae.config.shift_factor = 0

    # # 1. 导出 VAE 到 ONNX
    # vae_onnx_path = "vae.onnx"
    
    # if not os.path.exists(vae_trt_path):
    #     export_vae_to_onnx(vae, vae_onnx_path)

    # # 2. 转换 VAE 为 TensorRT
    # vae_trt_path = "vae.trt"
    # if not os.path.exists(vae_trt_path):
    #     build_vae_tensorrt_engine('vae.onnx', vae_trt_path)

    # # 5. 导出 denoising_unet 到 ONNX
    # denoising_unet, _ = UNet3DConditionModel.from_pretrained(
    #     OmegaConf.to_container(config.model),
    #     args.inference_ckpt_path,
    # ).to(torch.float16)

    unet_onnx_path = "./trt_engines/denoising_unet_onnx_wrapper_16/denoising_unet.onnx"
    # if not os.path.exists(unet_onnx_path):
    #     export_denoising_unet_to_onnx(denoising_unet, unet_onnx_path)

    # 6. 转换 denoising_unet 为 TensorRT
    unet_trt_path = "./trt_engines/denoising_unet_dynamic_10_2.trt"
    if not os.path.exists(unet_trt_path):
        build_unet_tensorrt_engine(unet_onnx_path, unet_trt_path)


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/stage2.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, default='./checkpoints/latentsync_unet.pt')
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)
    main(config, args)

    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f'Finish Processing at cost of time {elapsed_time}s')
