import requests

url = "http://localhost:8000/predict"

# 准备数据
files = {"image": open("./assets/cat-1.png", "rb")}
data = {"texts": "a photo of a cat,a photo of a dog,a photo of a bird"}

# 发送请求
response = requests.post(url, files=files, data=data)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print("预测结果:")
    for prediction in result["predictions"]:
        print(f"文本: {prediction['text']}")
        print(f"相似度分数: {prediction['similarity_score']:.4f}")
        print(f"概率: {prediction['probability']:.4f}")
    print(f"推理时间: {result['performance']['inference_time']}")
else:
    print(f"请求失败: {response.status_code}")
    print(response.text)