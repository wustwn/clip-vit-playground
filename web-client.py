import requests

url = "http://localhost:8000/predict"

# 准备数据
files = {"image": open("./assets/cat-1.png", "rb")}
data = {"texts": "a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo"}

# 发送请求
response = requests.post(url, files=files, data=data)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print("预测结果:")
    prediction = result["prediction"]
    print(f"相似度分数: {prediction['similarity_score']:.4f}")
    if prediction["probability"] is not None:
        print(f"概率: {prediction['probability']:.4f}")
else:
    print(f"请求失败: {response.status_code}")
    print(response.text)