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

# 初始化应用
app = FastAPI(
    title="CLIP Model API",
    description="API for OpenAI's CLIP model for image-text similarity",
    version="1.0.0"
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型和处理器
model = None
processor = None
device = None

@app.on_event("startup")
async def load_model():
    """在应用启动时加载模型"""
    global model, processor, device
    
    logger.info("正在加载模型...")
    start_time = time.time()
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型和处理器
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model.eval()  # 设置为评估模式
        
        load_time = time.time() - start_time
        logger.info(f"模型加载完成! 耗时: {load_time:.2f}秒")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise HTTPException(status_code=500, detail="模型加载失败")

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
    texts: str = "a photo of a cat,a photo of a dog",
    return_probs: bool = True
):
    """执行图像-文本相似度预测
    
    - **image**: 上传的图像文件 (JPEG, PNG)
    - **texts**: 逗号分隔的文本描述列表
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
        
        # 处理文本输入
        text_list = [text.strip() for text in texts.split(",")]
        
        # 预处理输入
        inputs = processor(
            text=text_list,
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
            "texts": text_list,
            "predictions": []
        }
        
        for i, text in enumerate(text_list):
            prediction = {
                "text": text,
                "similarity_score": float(logits_per_image[0, i].item()),
                "probability": float(probs[i]) if return_probs else None
            }
            response["predictions"].append(prediction)
        
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