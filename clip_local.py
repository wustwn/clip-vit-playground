from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

# 检测可用设备并选择GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载模型并移动到设备
start_load_model = time.time()
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
load_model_time = time.time() - start_load_model

# 加载处理器
start_load_processor = time.time()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
load_processor_time = time.time() - start_load_processor

# 加载图片
start_load_image = time.time()
image = Image.open("./assets/cat-1.png")
load_image_time = time.time() - start_load_image

# 预处理
start_preprocess = time.time()
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True
)
inputs = {k: v.to(device) for k, v in inputs.items()}
preprocess_time = time.time() - start_preprocess

# 使用PyTorch Profiler进行推理性能分析
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    with record_function("model_inference"):
        start_inference = time.time()
        outputs = model(**inputs)
        inference_time = time.time() - start_inference

# 后处理
start_postprocess = time.time()
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
postprocess_time = time.time() - start_postprocess

# 打印结果
print("概率分布:", probs.detach().cpu().numpy())

# 打印手动计时结果
print("\n===== 手动计时性能指标 =====")
print(f"模型加载时间: {load_model_time:.4f}秒")
print(f"处理器加载时间: {load_processor_time:.4f}秒")
print(f"图片加载时间: {load_image_time:.4f}秒")
print(f"预处理时间: {preprocess_time:.4f}秒")
print(f"推理时间: {inference_time:.4f}秒")
print(f"后处理时间: {postprocess_time:.4f}秒")
print("===========================")

# 打印PyTorch Profiler结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))