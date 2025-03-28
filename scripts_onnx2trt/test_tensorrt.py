import pycuda.driver as cuda
import pycuda.autoinit
import torch
import numpy as np
import tensorrt as trt
trt.init_libnvinfer_plugins(None, "")

class TRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_trt_engine(engine_path)
        self.context = self.engine.create_execution_context()

    def load_trt_engine(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def allocate_buffers(self, dynamic_input_shapes):
        bindings = []
        inputs = []
        outputs = []

        num_bindings = self.engine.num_io_tensors  # 适用于 TensorRT 10.2

        for i in range(num_bindings):
            tensor_name = self.engine.get_binding_name(i)  # 获取张量名称
            dtype = self.engine.get_binding_dtype(i)  # 获取数据类型

            # **处理动态输入**
            if self.engine.binding_is_input(i) and tensor_name in dynamic_input_shapes:
                shape = dynamic_input_shapes[tensor_name]
                self.context.set_binding_shape(i, shape)  # ✅ **正确设置动态输入形状**
            else:
                shape = self.context.get_binding_shape(i)  # ✅ **获取当前 shape**

            # **分配内存**
            size = trt.volume(shape) * np.dtype(trt.nptype(dtype)).itemsize
            device_mem = cuda.mem_alloc(size)
            bindings.append(int(device_mem))

            # **区分输入和输出**
            if self.engine.binding_is_input(i):
                inputs.append(device_mem)
            else:
                outputs.append(device_mem)

        return bindings, inputs, outputs

    def infer(self, denoising_unet_input, t, audio_embeds):
        
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
        noise_pred = np.empty((2, 4, 16, 32, 32), dtype=np.float16)
        cuda.memcpy_dtoh(noise_pred, outputs[0]) # 2 4 16 32 32
        noise_pred_cuda = torch.tensor(noise_pred, dtype=torch.float16, device='cuda')
        return noise_pred_cuda

trt_infer = TRTInference("./trt_engines/denoising_unet_dynamic_10_2.trt")
tmp = torch.load('denoising_unet_input.pt', weights_only=True)
denoising_unet_input = tmp['denoising_unet_input']
audio_embeds = tmp['audio_embeds']

t = torch.tensor([951], dtype=torch.int64, device="cuda")  # 时间步

try:
    noise_pred = trt_infer.infer(denoising_unet_input, t, audio_embeds)
    print(noise_pred)
except Exception as e:
    print(f"发生错误: {e}")
