import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import pickle
import requests
from io import BytesIO
from tqdm import tqdm
import os

# 初始化ResNet模型
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet101(pretrained=True)
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

# 定義檢查並載入已有的.pkl檔案
def load_existing_features(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}

# 定義儲存特徵的檔案路徑
feature_file_path = '/Users/laiwei/Desktop/91APP_Image_Coding/image_features.pkl'

# 如果已經存在特徵檔案，則載入現有的特徵
features_by_category = load_existing_features(feature_file_path)

# 設定要讀取的多個 Excel 檔案路徑
excel_files = [
    '/Users/laiwei/Desktop/91APP_Image_Coding/訓練資料集(鞋子).xlsx',
    '/Users/laiwei/Desktop/91APP_Image_Coding/訓練資料集(上衣).xlsx',
    '/Users/laiwei/Desktop/91APP_Image_Coding/訓練資料集(包包).xlsx'
]

# 逐個讀取和處理Excel文件
for excel_file in excel_files:
    df = pd.read_excel(excel_file)
    
    # 逐行處理每一筆數據，並顯示進度條
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {os.path.basename(excel_file)}"):
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
        except requests.exceptions.RequestException:
            print(f"無法下載此圖：{image_url}")
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
with open(feature_file_path, 'wb') as f:
    pickle.dump(features_by_category, f)

