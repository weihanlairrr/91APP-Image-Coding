import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import pickle
import requests
from io import BytesIO
from tqdm import tqdm

# 初始化ResNet模型
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # 去掉最後的全連接層，只提取特徵
resnet.eval().to(device)

# 定義圖片預處理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 定義提取特徵的函數
def get_image_features(image, model):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()
    return features

# 讀取資料集Excel
df = pd.read_excel('C:/Users/albertlai/Desktop/python/91APP-AI-Image-Embedding/訓練資料集.xlsx')

# 定義儲存特徵的結構：以品牌、商品分類為鍵，存放特徵的列表
features_by_category = {}

# 逐行處理Excel表格中的每一筆數據，並顯示進度條
for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Images"):
    image_url = row['URL']
    brand = row['品牌']
    category = row['商品分類']
    angle = row['角度']
    number = row['編號']

    # 從URL下載圖片
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # 確認請求成功
        image = Image.open(BytesIO(response.content)).convert('RGB')  # 將圖片內容轉換為PIL Image格式
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image {image_url}: {e}")
        continue  # 如果下載失敗，跳過該圖片

    # 提取圖片特徵
    image_features = get_image_features(image, resnet)

    # 構建品牌 -> 商品分類 -> [圖片特徵, 標籤] 的結構
    if brand not in features_by_category:
        features_by_category[brand] = {}
    
    if category not in features_by_category[brand]:
        features_by_category[brand][category] = {
            "all_features": [],  # 保存每張圖片的特徵
            "labeled_features": []  # 保存每張圖片的標籤
        }

    # 儲存同品牌、同商品分類下的每張圖片特徵
    features_by_category[brand][category]["all_features"].append(image_features)

    # 儲存同品牌、同商品分類、同角度、同編號的特徵
    features_by_category[brand][category]["labeled_features"].append({
        "features": image_features,  # 保存每張圖片的特徵向量
        "labels": {
            "brand": brand,
            "category": category,
            "angle": angle,
            "number": number
        }
    })

# 保存每張圖片的特徵以及商品分類的平均特徵到本地
with open('C:/Users/albertlai/Desktop/python/91APP-AI-Image-Embedding/image_features.pkl', 'wb') as f:
    pickle.dump(features_by_category, f)
