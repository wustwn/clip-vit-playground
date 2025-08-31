from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import io
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储模型和处理器
model = None
processor = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """模型生命周期管理 (替代弃用的 on_event)"""
    global model, processor, device
    
    # 加载模型 (启动时)
    logger.info("正在加载模型...")
    start_time = time.time()
    
    try:
        # 检测设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        # 加载模型和处理器
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=False)
        
        model.eval()  # 设置为评估模式
        
        load_time = time.time() - start_time
        logger.info(f"模型加载完成! 耗时: {load_time:.2f}秒")
        yield
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise
    
    # 清理资源 (关闭时)
    logger.info("清理模型资源...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("资源清理完成")

# 初始化应用（使用新的 lifespan 处理器）
app = FastAPI(
    title="CLIP Model API",
    description="API for OpenAI's CLIP model for image-text similarity",
    version="1.0.0",
    lifespan=lifespan  # 使用新的生命周期处理器
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    """健康检查端点"""
    return {
        "status": "running",
        "model": "openai/clip-vit-large-patch14",
        "device": str(device)
    }

@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="上传的图像文件"),
    text: str = "",
    return_probs: bool = True
):
    """执行图像-文本相似度预测
    
    - **image**: 上传的图像文件 (JPEG, PNG)
    - **text**: 文本描述
    - **return_probs**: 是否返回概率分布 (默认为True)
    """
    start_time = time.time()
    
    # 验证模型是否加载
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型未加载完成，请稍后再试")
    
    try:
        # 读取上传的图像
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # 预处理输入
        inputs = processor(
            text=text,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 执行推理
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 处理输出
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # 构建响应
        response = {
            "prediction": {
                "text": text,
                "similarity_score": float(logits_per_image[0, 0].item()),
                "probability": float(probs[0]) if return_probs else None
            }
        }
        
        # 添加性能指标
        inference_time = time.time() - start_time
        response["performance"] = {
            "inference_time": f"{inference_time:.4f}秒",
            "device": str(device)
        }
        
        return response
    
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)