#%% 導入區
import streamlit as st
import pandas as pd
import zipfile
import os
import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import pickle
import shutil
import numpy as np
import platform
from torchvision.models import ResNet50_Weights
import re

# 設定 Streamlit 頁面的標題和圖示
st.set_page_config(page_title='TP自動化編圖工具', page_icon='👕')

# 自定義 CSS 以調整頁面樣式
custom_css = """
<style>
.main {
    padding-left: 28%; 
    padding-right: 28%;
}
div.block-container{padding-top:4rem;
}
.stButton > button {
    padding: 5px 30px;
    background: #5A5B5E!important;
    color: #f5f5f5!important;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    margin: 5px 0;
}
.stDownloadButton button {
    background: #5A5B5E!important;
    color: #f5f5f5!important;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}
[data-testid='stFileUploader'] section button {
    color: #46474A !important;
    border-radius: 5px;
    border: none;
    padding: 5px 40px;
}
.stButton > button:hover  {
    background: #8A8B8D!important;
}
.stDownloadButton button:hover {
    background: #8A8B8D!important;
}
button:hover  {
    background: #D3D3D3!important;
}
</style>
"""

# 將自定義 CSS 應用到頁面
st.markdown(custom_css, unsafe_allow_html=True)

# 設定運行裝置，優先使用 GPU（CUDA），否則使用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入預訓練的 ResNet50 模型，並移除最後一層全連接層
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  
resnet.eval().to(device)

# 定義圖像預處理流程，包括調整大小、中心裁剪、轉換為張量及正規化
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

#%% 檔案與自訂參數

# 資料集檔名
train_file = "image_features.pkl"

# 檔名角度對照表
angle_filename_reference = "ADS檔名角度對照表.xlsx"

# 編圖的編號上限
label_limit = 10

# 讀取「移到外層的檔名」的第一欄作為關鍵字列表，並確保所有元素為字串
keywords_to_skip = pd.read_excel(angle_filename_reference, sheet_name='移到外層的檔名', usecols=[0]).iloc[:, 0].dropna().astype(str).tolist()

# 讀取「有條件使用的檔名」，拆分 set_a 和 set_b 的逗號分隔值
substitute_df = pd.read_excel(angle_filename_reference, sheet_name='有條件使用的檔名', usecols=[0, 1])
substitute = []
for _, row in substitute_df.iterrows():
    substitute.append({
        "set_a": row.iloc[0].split(','),  # 將 set_a 列表中的值以逗號分隔
        "set_b": row.iloc[1].split(',')   # 將 set_b 列表中的值以逗號分隔
    })

# 讀取「可以重複分配的角度」的第一欄作為可重複分配的角度列表
reassigned_allowed = pd.read_excel(angle_filename_reference, sheet_name='可以重複分配的角度', usecols=[0]).iloc[:, 0].dropna().tolist()

# 讀取「角度禁止規則」的前3欄並組裝成結構化字典
angle_banning_df = pd.read_excel(angle_filename_reference, sheet_name='角度禁止規則', usecols=[0, 1, 2])
angle_banning_rules = [
    {
        "if_appears_in_angle": row.iloc[0].split(','),  # 將條件角度分隔為列表
        "banned_angle": row.iloc[1],
        "banned_angle_logic": row.iloc[2]
    }
    for _, row in angle_banning_df.iterrows()
]

# 讀取「商品分類及關鍵字條件」的前3欄並組裝成字典
category_rules_df = pd.read_excel(angle_filename_reference, sheet_name='商品分類及關鍵字條件', usecols=[0, 1, 2])
category_rules = {
    row.iloc[0]: {
        "keywords": row.iloc[1].split(','),
        "match_all": row.iloc[2]
    }
    for _, row in category_rules_df.iterrows()
}

#%% function
def get_image_features(image, model):
    """
    提取圖像特徵的方法。
    參數:
        image: PIL.Image 對象，輸入的圖像
        model: 深度學習模型，用於提取特徵
    回傳:
        特徵向量（numpy 陣列）
    """
    image = preprocess(image).unsqueeze(0).to(device)  # 預處理並添加批次維度
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()  # 提取特徵並展平
    return features

def cosine_similarity(a, b):
    """
    計算兩個向量之間的餘弦相似度。
    參數:
        a, b: numpy 陣列，待比較的向量
    回傳:
        餘弦相似度值
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def reset_file_uploader():
    """
    重置文件上傳器的狀態，並刪除上傳的圖像和臨時壓縮檔。
    """
    st.session_state['file_uploader_key1'] += 1  # 增加 key 以重置上傳器
    if os.path.exists("uploaded_images"):
        shutil.rmtree("uploaded_images")  # 刪除上傳的圖像資料夾
    if os.path.exists("temp.zip"):
        os.remove("temp.zip")  # 刪除臨時壓縮檔

def unzip_file(uploaded_zip):
    """
    解壓上傳的壓縮檔，並處理解壓過程中的編碼問題。
    參數:
        uploaded_zip: 上傳的壓縮檔案
    """
    system = platform.system()  # 獲取作業系統名稱
    
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        for member in zip_ref.infolist():
            # 跳過系統自動生成的文件
            if "__MACOSX" in member.filename or member.filename.startswith('.'):
                continue
            
            # 根據不同的作業系統處理檔名編碼
            if system == "Windows":
                try:
                    member.filename = member.filename.encode('utf-8').decode('utf-8')
                except UnicodeDecodeError:
                    member.filename = member.filename.encode('utf-8').decode('latin1')
            elif system == "Darwin":
                try:
                    member.filename = member.filename.encode('cp437').decode('utf-8')
                except UnicodeDecodeError:
                    member.filename = member.filename.encode('cp437').decode('latin1')
            else:
                try:
                    member.filename = member.filename.encode('utf-8').decode('utf-8')
                except UnicodeDecodeError:
                    member.filename = member.filename.encode('utf-8').decode('latin1')
            
            # 解壓每個檔案到指定的資料夾
            zip_ref.extract(member, "uploaded_images")

def get_images_in_folder(folder_path):
    """
    獲取指定資料夾中的所有圖像檔案。
    參數:
        folder_path: 資料夾的路徑
    回傳:
        圖像檔案的相對路徑和完整路徑的列表
    """
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 跳過隱藏檔案和子目錄
            if file.startswith('.') or os.path.isdir(os.path.join(root, file)):
                continue
            # 檢查檔案副檔名是否為圖像格式
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                full_image_path = os.path.join(root, file)
                relative_image_path = os.path.relpath(full_image_path, folder_path)
                image_files.append((relative_image_path, full_image_path))
    return image_files

def rename_numbers_in_folder(results):
    """
    根據編號重新命名資料夾中的圖像檔案。
    參數:
        results: 圖像處理的結果列表
    回傳:
        更新後的結果列表
    """
    folders = set([result["資料夾"] for result in results])  # 獲取所有資料夾名稱
    for folder in folders:
        folder_results = [r for r in results if r["資料夾"] == folder]
        # 檢查是否有未編號的圖像
        if any(pd.isna(r["編號"]) or r["編號"] == "" for r in folder_results):
            continue
        # 按照編號排序
        folder_results.sort(key=lambda x: int(x["編號"]))
        for idx, result in enumerate(folder_results):
            if idx < label_limit:
                result["編號"] = f'{idx+1:02}'  # 編號格式為兩位數
            else:
                result["編號"] = "超過上限"  # 超過編號上限時標記
    return results

def rename_and_zip_folders(results, output_excel_data, skipped_images):
    """
    重新命名圖像檔案並壓縮處理後的資料夾和結果 Excel 檔。
    參數:
        results: 圖像處理的結果列表
        output_excel_data: 結果的 Excel 資料
        skipped_images: 被跳過的圖像列表
    回傳:
        壓縮檔的二進位數據
    """
    for result in results:
        folder_name = result["資料夾"]
        image_file = result["圖片"]
        new_number = result["編號"]
    
        folder_path = os.path.join("uploaded_images", folder_name)
        main_folder_path = os.path.join(folder_path, "1-Main")
        all_folder_path = os.path.join(main_folder_path, "All")
        os.makedirs(all_folder_path, exist_ok=True)  # 創建主資料夾和 All 資料夾
            
        old_image_path = os.path.join(folder_path, image_file)

        if new_number == "超過上限" or pd.isna(new_number):
            new_image_path = os.path.join(folder_path, os.path.basename(image_file))  # 將圖片移動到外層資料夾
        else:
            new_image_name = f"{folder_name}_{new_number}.jpg"  # 新的圖像名稱
            new_image_path = os.path.join(all_folder_path, new_image_name)
        
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

        if os.path.exists(old_image_path):
            os.rename(old_image_path, new_image_path)  # 重新命名或移動圖像檔案

    for skipped_image in skipped_images:
        folder_name = skipped_image["資料夾"]
        image_file = skipped_image["圖片"]
        folder_path = os.path.join("uploaded_images", folder_name)
        old_image_path = os.path.join(folder_path, image_file)
        
        if os.path.exists(old_image_path):
            new_image_path = os.path.join(folder_path, os.path.basename(image_file))  # 將圖片移動到外層資料夾
            os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
            os.rename(old_image_path, new_image_path)  # 移動圖片

    zip_buffer = BytesIO()  # 創建內存中的緩衝區
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder in os.listdir("uploaded_images"):
            folder_path = os.path.join("uploaded_images", folder)
            if os.path.isdir(folder_path):
                new_folder_name = f"{folder}_OK"  # 新的資料夾名稱
                new_folder_path = os.path.join("uploaded_images", new_folder_name)
                os.rename(folder_path, new_folder_path)  # 重新命名資料夾
                
                for root, dirs, files in os.walk(new_folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, "uploaded_images"))  # 添加檔案到壓縮檔
        
        zipf.writestr("編圖結果.xlsx", output_excel_data)  # 添加結果 Excel 檔到壓縮檔

    return zip_buffer.getvalue()  # 返回壓縮檔的二進位數據

def category_match(image_files, keywords, match_all):
    """
    根據給定的條件判斷資料夾是否符合特定商品分類。
    參數:
        image_files: 資料夾中所有圖像檔案名稱
        keywords: 判斷所需的關鍵字列表
        match_all: 布林值，指示是否需要所有關鍵字都存在 (True) 還是只需任一關鍵字存在 (False)
    回傳:
        布林值，指示資料夾是否符合該商品分類
    """
    if match_all:
        return all(any(keyword in image_file for image_file in image_files) for keyword in keywords)
    else:
        return any(any(keyword in image_file for image_file in image_files) for keyword in keywords)
    
#%% 主函數

# 從 pickle 檔案中載入圖像特徵數據，並保存原始資料以供後續重置
with open(train_file, 'rb') as f:
    features_by_category = pickle.load(f)
    original_features_by_category = {k: v.copy() for k, v in features_by_category.items()}

# 初始化 session_state 中的文件上傳器 key
if 'file_uploader_key1' not in st.session_state:
    st.session_state['file_uploader_key1'] = 0

# 設定頁面標題
st.header("TP 編圖工具")
st.write("\n")

# 創建文件上傳器，允許上傳 zip 檔案
uploaded_zip = st.file_uploader(
    "上傳 zip 檔案", 
    type=["zip"], 
    key='file_uploader_' + str(st.session_state['file_uploader_key1'])
)

# 創建佔位符以動態顯示選擇框和按鈕
selectbox_placeholder = st.empty()
button_placeholder = st.empty()

if uploaded_zip:
    with selectbox_placeholder:
        selected_brand = st.selectbox(
            "請選擇品牌", 
            list(features_by_category.keys())  # 從載入的特徵數據中獲取品牌列表
        )
    with button_placeholder:
        start_running = st.button("開始執行")  # 開始執行按鈕

if uploaded_zip and start_running:
    # 清空選擇框和按鈕的佔位符
    selectbox_placeholder.empty()
    button_placeholder.empty()
    st.write("\n")
    
    # 如果已存在上傳的圖像資料夾，則刪除
    if os.path.exists("uploaded_images"):
        shutil.rmtree("uploaded_images")
        
    # 將上傳的 zip 檔案寫入臨時檔案
    with open("temp.zip", "wb") as f:
        f.write(uploaded_zip.getbuffer())

    # 解壓上傳的 zip 檔案
    unzip_file("temp.zip")

    # 初始化特殊映射字典
    special_mappings = {}
    if selected_brand == "ADS":
        # 讀取特定品牌的檔名角度對照表
        df_angles = pd.read_excel(angle_filename_reference, sheet_name="檔名角度對照表")
        for idx, row in df_angles.iterrows():
            keyword = str(row.iloc[0]).strip()
            category_raw = str(row.iloc[1]).strip()
            if category_raw == 'nan' or category_raw == '':
                category = None
                category_filename = None
            else:
                # 使用正則表達式解析商品分類
                match = re.match(r'^(.*)\((.*)\)$', category_raw)
                if match:
                    category = match.group(1).strip()
                    category_filename_raw = match.group(2).strip()
                    category_filename = [x.strip() for x in category_filename_raw.split(',')]  # 修改此處，支持多個條件
                else:
                    category = category_raw
                    category_filename = None
            angle = str(row.iloc[2]).strip()
            angles = [a.strip() for a in angle.split(',')]
            special_mappings[keyword] = {
                'category': category, 
                'category_filename': category_filename,
                'angles': angles
            }

    # 獲取所有上傳的圖像資料夾
    image_folders = [
        f for f in os.listdir("uploaded_images") 
        if os.path.isdir(os.path.join("uploaded_images", f)) 
        and not f.startswith('__MACOSX') and not f.startswith('.')
    ]
    results = []  # 存儲處理結果
    skipped_images = []  # 存儲被跳過的圖像
    progress_bar = st.progress(0)  # 創建進度條
    progress_text = st.empty()  # 創建進度文字

    total_folders = len(image_folders)  # 總資料夾數量
    processed_folders = 0  # 已處理的資料夾數量

    # set_b 只有在 set_a 不存在時才能使用，否則需要被移到外層資料夾
    group_conditions = substitute

    # 遍歷每個圖像資料夾進行處理
    for folder in image_folders:
        # 每次處理新資料夾前，重置 features_by_category
        features_by_category = {k: v.copy() for k, v in original_features_by_category.items()}

        folder_path = os.path.join("uploaded_images", folder)
        image_files = get_images_in_folder(folder_path)  # 獲取資料夾中的圖像檔案
        if not image_files:
            st.warning(f"資料夾 {folder} 中沒有有效的圖片，跳過此資料夾")
            continue
        folder_features = []  # 存儲資料夾中所有圖像的特徵

        progress_text.text(f"正在處理資料夾: {folder}")  # 更新進度文字

        special_images = []  # 存儲特殊映射的圖像
        folder_special_category = None  # 存儲資料夾的特殊分類

        # 初始化每組條件的存在標記
        group_presence = []
        for group in group_conditions:
            group_presence.append({
                "set_a_present": False,
                "set_b_present": False
            })

        # 檢查每個分組條件是否在資料夾中存在
        for image_file, image_path in image_files:
            if image_file.startswith('.') or os.path.isdir(image_path):
                continue

            for idx, group in enumerate(group_conditions):
                if any(substr in image_file for substr in group["set_a"]):
                    group_presence[idx]["set_a_present"] = True
                if any(substr in image_file for substr in group["set_b"]):
                    group_presence[idx]["set_b_present"] = True

        image_filenames = [img[0] for img in image_files]  # 獲取所有圖像檔案名稱

        # 遍歷每個圖像檔案進行特徵提取和分類
        for image_file, image_path in image_files:
            if image_file.startswith('.') or os.path.isdir(image_path):
                continue

            # 檢查圖像檔名是否包含需要跳過的關鍵字
            if any(keyword in image_file for keyword in keywords_to_skip):
                skipped_images.append({
                    "資料夾": folder, 
                    "圖片": image_file
                })
                continue

            skip_image = False
            # 根據分組條件決定是否跳過圖像
            for idx, group in enumerate(group_conditions):
                if any(substr in image_file for substr in group["set_b"]):
                    if group_presence[idx]["set_a_present"] and group_presence[idx]["set_b_present"]:
                        skipped_images.append({
                            "資料夾": folder, 
                            "圖片": image_file
                        })
                        skip_image = True
                        break

            if skip_image:
                continue

            special_angles = []
            special_category = None
            category_filename = None
            # 檢查是否有特殊映射
            if special_mappings:
                for substr, mapping in special_mappings.items():
                    if substr in image_file:
                        special_angles = mapping['angles']
                        special_category = mapping['category']
                        category_filename = mapping.get('category_filename')
                        if category_filename:
                            # 修改此處，支持多個 category_filename 條件
                            if any(cond in fname for fname in image_filenames for cond in category_filename):
                                pass 
                            else:
                                special_category = None 
                        if special_category and not folder_special_category:
                            folder_special_category = special_category
                        break

            img = Image.open(image_path).convert('RGB')  # 打開並轉換圖像為 RGB 模式
            img_features = get_image_features(img, resnet)  # 提取圖像特徵
            folder_features.append({
                "image_file": image_file,
                "features": img_features,
                "special_angles": special_angles,
                "special_category": special_category
            })

            if special_angles:
                special_images.append({
                    "image_file": image_file,
                    "special_angles": special_angles
                })

        best_category = None  # 初始化最佳分類

        if len(folder_features) == 0:
            st.warning(f"資料夾 {folder} 中沒有有效的圖片，跳過此資料夾")
            continue

        # 資料夾必須含有以下檔名，才能夠被分配到指定的商品分類
        for category, rule in category_rules.items():
            if category in features_by_category[selected_brand]:
                if not category_match([file[0] for file in image_files], rule["keywords"], rule["match_all"]):
                    features_by_category[selected_brand].pop(category, None)

        # 如果有特殊分類，則設定為最佳分類
        if folder_special_category:
            best_category = {
                'brand': selected_brand, 
                'category': folder_special_category
            }
        else:
            # 計算每個分類的相似度，選擇相似度最高的分類
            category_similarities = {}
            for img_data in folder_features:
                img_features = img_data["features"]
            
                for brand in features_by_category:
                    for category in features_by_category[brand]:
                        image_similarities = []
                        
                        # 計算該圖像與分類中每個標籤特徵的相似度
                        for item in features_by_category[brand][category]["labeled_features"]:
                            item_features = item["features"]
                            similarity = cosine_similarity(img_features, item_features)
                            image_similarities.append(similarity)
                        
                        # 將相似度排序，取前3個最高的相似度（如果圖片不足3張則取全部）
                        top_similarities = sorted(image_similarities, reverse=True)[:3]
                        avg_top_similarity = sum(top_similarities) / len(top_similarities)
                        
                        # 儲存分類的平均相似度
                        if category not in category_similarities:
                            category_similarities[category] = []
                        category_similarities[category].append(avg_top_similarity)
            
            # 計算每個分類的平均相似度並選擇最高的分類
            best_category = None
            highest_avg_similarity = -1
            for category, similarities in category_similarities.items():
                folder_avg_similarity = sum(similarities) / len(similarities)
                if folder_avg_similarity > highest_avg_similarity:
                    highest_avg_similarity = folder_avg_similarity
                    best_category = {
                        'brand': selected_brand, 
                        'category': category
                    }

        # 根據最佳分類獲取相關的標籤和編號
        filtered_by_category = features_by_category[selected_brand][
            best_category["category"]
        ]["labeled_features"]

        angle_to_number = {
            item["labels"]["angle"]: item["labels"]["number"] 
            for item in filtered_by_category
        }

        used_angles = set()  # 已使用的角度集合
        final_results = {}  # 最終結果字典

        # 初始化規則標誌
        rule_flags = [False for _ in angle_banning_rules]

        # 遍歷每個圖像資料進行角度分配
        for img_data in folder_features:
            image_file = img_data["image_file"]
            special_angles = img_data["special_angles"]
            special_category = img_data["special_category"]
            img_features = img_data["features"]
            best_angle = None

            if special_angles:
                # 過濾有效的特殊角度
                valid_special_angles = [
                    angle for angle in special_angles 
                    if angle in angle_to_number
                ]
                if valid_special_angles:
                    if len(valid_special_angles) > 1:
                        best_angle = None
                        valid_angles_by_similarity = []
                        
                        # 根據相似度選擇最佳角度
                        for angle in valid_special_angles:
                            max_similarity = -1
                            for item in filtered_by_category:
                                if item["labels"]["angle"] == angle:
                                    sample_features = item["features"]
                                    similarity = cosine_similarity(
                                        img_features, sample_features
                                    )
                                    if similarity > max_similarity:
                                        max_similarity = similarity
                            
                            valid_angles_by_similarity.append(
                                (angle, max_similarity)
                            )
                        
                        # 根據相似度排序
                        valid_angles_by_similarity.sort(
                            key=lambda x: x[1], reverse=True
                        )
                        
                        for angle, similarity in valid_angles_by_similarity:
                            if angle not in reassigned_allowed and angle in used_angles:
                                pass
                            else:
                                best_angle = angle
                                best_similarity = similarity
                                break
                    
                        if best_angle:
                            used_angles.add(best_angle)  # 標記角度為已使用
                            label_info = {
                                "資料夾": folder,
                                "圖片": image_file,
                                "商品分類": best_category["category"],
                                "角度": best_angle,
                                "編號": angle_to_number[best_angle],
                                "最大相似度": f"{best_similarity * 100:.2f}%"
                            }
                            final_results[image_file] = label_info
                            # 更新規則標誌
                            for idx, rule in enumerate(angle_banning_rules):
                                if best_angle in rule["if_appears_in_angle"]:
                                    rule_flags[idx] = True
                        else:
                            st.warning(
                                f"圖片 '{image_file}' 沒有可用的角度可以分配"
                            )
                            final_results[image_file] = None
                    else:
                        # 只有一個有效的特殊角度
                        special_angle = valid_special_angles[0]
                        if special_angle not in reassigned_allowed and special_angle in used_angles:
                            st.warning(
                                f"角度 '{special_angle}' 已被使用，圖片 '{image_file}' 無法分配"
                            )
                            final_results[image_file] = None
                        else:
                            used_angles.add(special_angle)  # 標記角度為已使用
                            label_info = {
                                "資料夾": folder,
                                "圖片": image_file,
                                "商品分類": best_category["category"],
                                "角度": special_angle,
                                "編號": angle_to_number[special_angle],
                                "最大相似度": "100.00%"
                            }
                            final_results[image_file] = label_info
                            # 更新規則標誌
                            for idx, rule in enumerate(angle_banning_rules):
                                if special_angle in rule["if_appears_in_angle"]:
                                    rule_flags[idx] = True
                else:
                    st.warning(
                        f"商品分類 '{best_category['category']}' 中沒有角度 '{', '.join(special_angles)}'，圖片 '{image_file}' 無法分配"
                    )
                    final_results[image_file] = None
            else:
                final_results[image_file] = None  # 非特殊圖像暫時不分配

        # 獲取所有非特殊的圖像
        non_special_images = [
            img_data for img_data in folder_features 
            if not img_data["special_angles"]
        ]

        if not special_mappings:
            non_special_images = folder_features  # 如果沒有特殊映射，所有圖像都是非特殊的

        image_similarity_store = {}

        # 定義函數檢查角度是否被禁止
        def is_banned_angle(item_angle, rule_flags):
            for idx, rule in enumerate(angle_banning_rules):
                if rule_flags[idx]:
                    if rule["banned_angle_logic"] == "等於":
                        if item_angle == rule["banned_angle"]:
                            return True
                    elif rule["banned_angle_logic"] == "包含":
                        if rule["banned_angle"] in item_angle:
                            return True
            return False

        # 計算非特殊圖像與標籤的相似度
        for img_data in non_special_images:
            image_file = img_data["image_file"]
            if final_results.get(image_file) is not None:
                continue

            img_features = img_data["features"]
            image_similarity_list = []
            for item in filtered_by_category:
                item_angle = item["labels"]["angle"]

                if is_banned_angle(item_angle, rule_flags):
                    continue

                item_features = item["features"]
                similarity = cosine_similarity(
                    img_features, item_features
                )

                image_similarity_list.append({
                    "image_file": image_file,
                    "similarity": similarity,
                    "label": item["labels"],
                    "folder": folder
                })

            # 根據相似度排序
            image_similarity_list.sort(
                key=lambda x: x["similarity"], reverse=True
            )
            unique_labels = []
            for candidate in image_similarity_list:
                if candidate["label"]["angle"] not in [
                    label["label"]["angle"] for label in unique_labels
                ]:
                    unique_labels.append(candidate)
                if len(unique_labels) == label_limit:
                    break

            image_similarity_store[image_file] = unique_labels

        unassigned_images = set(image_similarity_store.keys())  # 未分配的圖像集合

        # 進行角度分配，直到所有未分配的圖像都處理完
        while unassigned_images:
            angle_to_images = {}
            image_current_choices = {}
            
            for image_file in unassigned_images:
                similarity_list = image_similarity_store[image_file]
                candidate = None
                for candidate_candidate in similarity_list:
                    candidate_angle = candidate_candidate["label"]["angle"]

                    if is_banned_angle(candidate_angle, rule_flags):
                        continue
                    
                    if candidate_angle in reassigned_allowed or candidate_angle not in used_angles:
                        candidate = candidate_candidate
                        break
                else:
                    candidate = None
                    candidate_angle = None
                
                if candidate:
                    candidate_angle = candidate["label"]["angle"]
                    image_current_choices[image_file] = candidate
                    if candidate_angle not in angle_to_images:
                        angle_to_images[candidate_angle] = []
                    angle_to_images[candidate_angle].append(image_file)
            
            assigned_in_this_round = set()
            for angle, images in angle_to_images.items():
                if angle in reassigned_allowed:
                    for image_file in images:
                        candidate = image_current_choices[image_file]
                        final_results[image_file] = {
                            "資料夾": candidate["folder"],
                            "圖片": image_file,
                            "商品分類": candidate["label"]["category"],
                            "角度": angle,
                            "編號": candidate["label"]["number"],
                            "最大相似度": f"{candidate['similarity'] * 100:.2f}%"
                        }
                        assigned_in_this_round.add(image_file)
                elif len(images) == 1:
                    image_file = images[0]
                    candidate = image_current_choices[image_file]
                    final_results[image_file] = {
                        "資料夾": candidate["folder"],
                        "圖片": image_file,
                        "商品分類": candidate["label"]["category"],
                        "角度": angle,
                        "編號": candidate["label"]["number"],
                        "最大相似度": f"{candidate['similarity'] * 100:.2f}%"
                    }
                    used_angles.add(angle)  # 標記角度為已使用
                    assigned_in_this_round.add(image_file)
                else:
                    max_similarity = -1
                    best_image = None
                    for image_file in images:
                        candidate = image_current_choices[image_file]
                        if candidate['similarity'] > max_similarity:
                            max_similarity = candidate['similarity']
                            best_image = image_file
                    candidate = image_current_choices[best_image]
                    final_results[best_image] = {
                        "資料夾": candidate["folder"],
                        "圖片": best_image,
                        "商品分類": candidate["label"]["category"],
                        "角度": angle,
                        "編號": candidate["label"]["number"],
                        "最大相似度": f"{candidate['similarity'] * 100:.2f}%"
                    }
                    used_angles.add(angle)  # 標記角度為已使用
                    assigned_in_this_round.add(best_image)

            unassigned_images -= assigned_in_this_round  # 更新未分配的圖像
            if not assigned_in_this_round:
                break  # 如果沒有圖像在本輪被分配，則退出循環

        # 將最終分配結果添加到結果列表
        for image_file, assignment in final_results.items():
            if assignment is not None:
                results.append(assignment)

        processed_folders += 1
        progress_bar.progress(processed_folders / total_folders)  # 更新進度條

    # 清空進度條和進度文字
    progress_bar.empty()
    progress_text.empty()

    # 根據編號重新命名圖像
    results = rename_numbers_in_folder(results)

    # 將結果轉換為 DataFrame 並顯示在頁面上
    result_df = pd.DataFrame(results)
    st.dataframe(result_df, hide_index=True, use_container_width=True)

    # 將結果 DataFrame 寫入 Excel 檔案
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, index=False)
    excel_data = excel_buffer.getvalue()

    # 重新命名並壓縮資料夾和結果 Excel 檔案
    zip_data = rename_and_zip_folders(results, excel_data, skipped_images)
    
    # 刪除上傳的圖像資料夾和臨時壓縮檔
    shutil.rmtree("uploaded_images")
    os.remove("temp.zip") 
    
    # 提供下載按鈕，下載處理後的壓縮檔
    if st.download_button(
        label="下載編圖結果",
        data=zip_data,
        file_name="編圖結果.zip",
        mime="application/zip",
        on_click=reset_file_uploader
    ):
        st.rerun()  # 下載後重新運行應用以重置狀態
