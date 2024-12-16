import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import pickle
import requests
from io import BytesIO
from tqdm import tqdm
import os
from torchvision.models import ResNet50_Weights

# 檢查是否可以使用 MPS（Apple GPU）
if torch.backends.mps.is_available():
    device = "mps"
    print("正在使用 Apple MPS 加速")
elif torch.cuda.is_available():
    device = "cuda"
    print("正在使用 CUDA 加速")
else:
    device = "cpu"
    print("正在使用 CPU")

weights = ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights)
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
        features = model(image).detach().cpu().numpy().flatten()
    return features

# 定義檢查並載入已有的.pkl檔案
def load_existing_features(file_path):
    if os.path.exists(file_path):
        print("使用既有的訓練資料")
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}

# 定義儲存特徵的檔案路徑
feature_file_path = '/Users/laiwei/Desktop/91APP_Image_Coding/dependencies/image_features.pkl'

# 如果已經存在特徵檔案，則載入現有的特徵
features_by_category = load_existing_features(feature_file_path)

# 設定要讀取的多個 Excel 檔案路徑
excel_files = [
    '/Users/laiwei/Desktop/91APP_Image_Coding/資料集/鞋子.xlsx',
    '/Users/laiwei/Desktop/91APP_Image_Coding/資料集/上身.xlsx',
    '/Users/laiwei/Desktop/91APP_Image_Coding/資料集/下身.xlsx',
    '/Users/laiwei/Desktop/91APP_Image_Coding/資料集/套裝.xlsx',
    '/Users/laiwei/Desktop/91APP_Image_Coding/資料集/雙面外套.xlsx',
    '/Users/laiwei/Desktop/91APP_Image_Coding/資料集/三合一外套.xlsx',
    '/Users/laiwei/Desktop/91APP_Image_Coding/資料集/包包.xlsx',
    '/Users/laiwei/Desktop/91APP_Image_Coding/資料集/帽子.xlsx',
    '/Users/laiwei/Desktop/91APP_Image_Coding/資料集/襪子.xlsx',
    '/Users/laiwei/Desktop/91APP_Image_Coding/資料集/其他配件.xlsx',
]

# 逐個讀取和處理 Excel 文件
for excel_file in excel_files:
    df = pd.read_excel(excel_file)
    
    # 逐行處理每一筆數據，並顯示進度條
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {os.path.basename(excel_file)}"):
        image_url = row['URL']
        brand = row['品牌']
        category = row['商品分類']
        angle = row['角度']
        number = row['編號']

        # 從 URL 下載圖片
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except requests.exceptions.RequestException:
            print(f"無法下載此圖：{image_url}")
            continue

        # 提取圖片特徵
        image_features = get_image_features(image, resnet)

        # 構建品牌 -> 商品分類 -> [圖片特徵, 標籤] 的結構
        if brand not in features_by_category:
            features_by_category[brand] = {}
        
        if category not in features_by_category[brand]:
            features_by_category[brand][category] = {
                "labeled_features": []
            }

        # 儲存同品牌、同商品分類、同角度、同編號的特徵
        features_by_category[brand][category]["labeled_features"].append({
            "features": image_features,
            "labels": {
                "brand": brand,
                "category": category,
                "angle": angle,
                "number": number
            }
        })

# 儲存每張圖片的特徵到本地
with open(feature_file_path, 'wb') as f:
    pickle.dump(features_by_category, f)

# 移除重複特徵的函數
def remove_duplicates(features_by_category):
    deleted_count = 0
    deleted_by_category = {}
    for brand, categories in features_by_category.items():
        for category, data in categories.items():
            unique_entries = set()
            unique_labeled_features = []

            for item in data["labeled_features"]:
                feature_tuple = (
                    tuple(item["features"]),
                    item["labels"]["brand"],
                    item["labels"]["category"],
                    item["labels"]["angle"]
                )

                if feature_tuple not in unique_entries:
                    unique_entries.add(feature_tuple)
                    unique_labeled_features.append(item)
                else:
                    deleted_count += 1
                    if category not in deleted_by_category:
                        deleted_by_category[category] = 0
                    deleted_by_category[category] += 1

            features_by_category[brand][category]["labeled_features"] = unique_labeled_features
    return features_by_category, deleted_count, deleted_by_category

# 執行去重並重新儲存
features_by_category, deleted_count, deleted_by_category = remove_duplicates(features_by_category)
with open(feature_file_path, 'wb') as f:
    pickle.dump(features_by_category, f)

# 列出刪除結果
print(f"總共刪除了 {deleted_count} 筆重複資料")
for category, count in deleted_by_category.items():
    print(f"{category}: {count} 筆")
