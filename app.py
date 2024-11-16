#%% 導入區
import streamlit as st
import pandas as pd
import zipfile
import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageOps
from io import BytesIO
import pickle
import shutil
import numpy as np
import re
import tempfile
from collections import Counter
import chardet
import faiss  
import multiprocessing

st.set_page_config(page_title='TP自動化編圖工具', page_icon='👕')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
faiss.omp_set_num_threads(multiprocessing.cpu_count())

# 自定義 CSS 以調整頁面樣式
custom_css = """
<style>
div.stTextInput > label {
    display: none;
}   
div.block-container {
    padding-top: 3rem;
}
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    padding: 5px 30px;
    background: #5A5B5E !important;
    color: #f5f5f5 !important;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    margin: 5px 0;
}
.stDownloadButton button {
    background: #5A5B5E !important;
    color: #f5f5f5 !important;
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
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background: #8A8B8D !important;
}
.stDownloadButton button:hover {
    background: #8A8B8D !important;
}
button:hover {
    background: #D3D3D3 !important;
}
</style>

"""

# 將自定義 CSS 應用到頁面
st.markdown(custom_css, unsafe_allow_html=True)

#%% function
@st.cache_resource
def load_resnet_model():
    """
    懶加載 ResNet 模型，並移除最後一層。
    """
    device = torch.device("cpu")
    weights_path = "dependencies/resnet50.pt"
    resnet = models.resnet50()
    resnet.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval().to(device)
    return resnet

@st.cache_resource
def get_preprocess_transforms():
    """
    定義圖像預處理流程，包括調整大小、中心裁剪、轉換為張量及正規化。
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.44, 0.406],
            std=[0.2, 0.2, 0.2]
        ),
    ])

@st.cache_resource
def load_image_features(train_file_path):
    """
    懶加載 .pkl 檔案，並返回特徵資料。
    參數:
        train_file_path: .pkl 檔案的路徑
    回傳:
        特徵資料字典
    """
    with open(train_file_path, 'rb') as f:
        features_by_category = pickle.load(f)
    return features_by_category

def initialize_session_state():
    """
    初始化所有 session_state 變數。
    """
    default_values = {
        'file_uploader_key1': 0,
        'file_uploader_key2': 4,
        'filename_changes': {},
        'confirmed_changes': {},
        'image_cache': {},
        'folder_values': {},
        'has_duplicates': False,
        'duplicate_filenames': [],
        'previous_selected_folder': None
    }

    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_image_features(image, model):
    """
    提取圖像特徵的方法，支援 macOS MPS、CUDA 和 CPU。
    參數:
        image: PIL.Image 對象，輸入的圖像
        model: 深度學習模型，用於提取特徵
    回傳:
        特徵向量（numpy 陣列）
    """
    # 根據設備設定運行裝置
    device = torch.device("cpu")
    image = preprocess(image).unsqueeze(0).to(device)  # 預處理並添加批次維度
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()  # 提取特徵並展平
    return features

def l2_normalize(vectors):
    """
    對向量進行 L2 正規化。
    參數:
        vectors: 2D numpy array (n_samples, n_features)
    回傳:
        正規化後的向量
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def reset_file_uploader():
    """
    重置文件上傳器的狀態，並刪除上傳的圖像和臨時壓縮檔。
    """
    st.session_state['file_uploader_key1'] += 1 
    st.session_state['file_uploader_key2'] += 1
    st.session_state['filename_changes'].clear()  # 清空檔名變更的緩存

def unzip_file(uploaded_zip):
    """
    解壓上傳的壓縮檔，並根據檔名自動偵測編碼。
    參數:
        uploaded_zip: 上傳的壓縮檔案
    """
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        for member in zip_ref.infolist():
            # 跳過系統自動生成的文件
            if "__MACOSX" in member.filename or member.filename.startswith('.'):
                continue
            
            # 使用 chardet 偵測檔名的編碼
            raw_bytes = member.filename.encode('utf-8', errors='ignore')  # 轉成 byte 格式以利編碼檢測
            detected_encoding = chardet.detect(raw_bytes)['encoding']
            
            try:
                # 使用偵測到的編碼解碼檔名
                member.filename = raw_bytes.decode(detected_encoding, errors='ignore')
            except (UnicodeDecodeError, LookupError, TypeError):
                # 如果偵測失敗，則使用 UTF-8 編碼並忽略錯誤
                member.filename = raw_bytes.decode('utf-8', errors='ignore')
            
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
    output_folder_path = "uploaded_images"  # 根資料夾
    
    for result in results:
        folder_name = result["資料夾"]
        image_file = result["圖片"]
        new_number = result["編號"]
        
        # 設定主資料夾路徑
        folder_path = os.path.join(output_folder_path, folder_name)
        main_folder_path = os.path.join(folder_path, main_folder_structure)
        os.makedirs(main_folder_path, exist_ok=True)  # 創建主資料夾
            
        old_image_path = os.path.join(folder_path, image_file)

        # 如果編號超過上限或為空，將圖片保留在最外層資料夾
        if new_number == "超過上限" or pd.isna(new_number):
            new_image_path = os.path.join(folder_path, os.path.basename(image_file))  
        else:
            new_image_name = f"{folder_name}_{new_number}.jpg"  # 新的圖像名稱
            new_image_path = os.path.join(main_folder_path, new_image_name)
        
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

        if os.path.exists(old_image_path):
            os.rename(old_image_path, new_image_path)  # 重新命名或移動圖像檔案

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

def generate_image_type_statistics(results):
    """
    根據分配結果生成每個資料夾的圖片類型統計。
    參數:
        results: 分配的結果列表
    回傳:
        統計結果的 DataFrame
    """
    statistics = []
    for folder, folder_results in results.groupby("資料夾"):
        # 統計含有"模特"的角度數量
        model_count = folder_results["角度"].str.contains("模特").sum()
        
        # 統計符合"細節"的角度
        detail_count = folder_results["角度"].apply(
            lambda x: any(key in x for key in ["細節", "D1", "D2", "D3", "D4", "D5", "H1", "H2", "H3", "H4", "H5"])
        ).sum()
        
        # 統計符合"平拍"的角度，排除 HM1-HM10
        excluded_angles = {"HM1", "HM2", "HM3", "HM4", "HM5", "HM6", "HM7", "HM8", "HM9", "HM10"}
        flat_lay_count = folder_results["角度"].apply(
            lambda x: x not in excluded_angles and "模特" not in x and not any(key in x for key in ["細節", "D1", "D2", "D3", "D4", "D5", "H1", "H2", "H3", "H4", "H5"])
        ).sum()
        
        # 儲存資料夾的統計結果
        statistics.append({
            "資料夾": folder,
            "模特": model_count,
            "平拍": flat_lay_count,
            "細節": detail_count
        })
    
    return pd.DataFrame(statistics)

#%% 自動編圖介面
tab1, tab2 = st.tabs([" 自動編圖 ", " 編圖複檢 "])
with tab1:
    resnet = load_resnet_model()
    preprocess = get_preprocess_transforms()

    brand_dependencies = {
        "ADS": {
            "train_file": "dependencies/image_features.pkl",
            "angle_filename_reference": "dependencies/ADS檔名角度對照表.xlsx",
            "label_limit": 10,
            "main_folder_structure": "1-Main/All"
        },
    }
    
    initialize_session_state() 
    brand_list = list(brand_dependencies.keys())
    st.write("\n")
    uploaded_zip = st.file_uploader(
        "上傳 Zip 檔案", 
        type=["zip"], 
        key='file_uploader_' + str(st.session_state['file_uploader_key1'])
    )

        
    if uploaded_zip:
        col1,col2,col3 = st.columns([1.5,2,2],vertical_alignment="center",gap="medium")
        selectbox_placeholder = col1.empty()
        button_placeholder = col2.empty()
        with selectbox_placeholder:
            selected_brand = st.selectbox(
                "請選擇品牌", brand_list, label_visibility="collapsed")
        with button_placeholder:
            start_running = st.button("開始執行")  

#%% 自動編圖邏輯            
        dependencies = brand_dependencies[selected_brand]
        train_file = dependencies["train_file"]
        angle_filename_reference = dependencies["angle_filename_reference"]
        label_limit = dependencies["label_limit"]
        main_folder_structure = dependencies["main_folder_structure"]
        
        keywords_to_skip = pd.read_excel(angle_filename_reference, sheet_name='移到外層的檔名', usecols=[0]).iloc[:, 0].dropna().astype(str).tolist()
        
        substitute_df = pd.read_excel(angle_filename_reference, sheet_name='有條件使用的檔名', usecols=[0, 1])
        substitute = [{"set_a": row.iloc[0].split(','), "set_b": row.iloc[1].split(',')} for _, row in substitute_df.iterrows()]
        
        reassigned_allowed = pd.read_excel(angle_filename_reference, sheet_name='可以重複分配的角度', usecols=[0]).iloc[:, 0].dropna().tolist()
        
        angle_banning_df = pd.read_excel(angle_filename_reference, sheet_name='角度禁止規則', usecols=[0, 1, 2])
        angle_banning_rules = [{"if_appears_in_angle": row.iloc[0].split(','), "banned_angle": row.iloc[1], "banned_angle_logic": row.iloc[2]} for _, row in angle_banning_df.iterrows()]
        
        category_rules_df = pd.read_excel(angle_filename_reference, sheet_name='商品分類及關鍵字條件', usecols=[0, 1, 2])
        category_rules = {row.iloc[0]: {"keywords": row.iloc[1].split(','), "match_all": row.iloc[2]} for _, row in category_rules_df.iterrows()}
        
        features_by_category = load_image_features(train_file)
        # 複製一份原始特徵資料，避免在處理時修改到原始資料
        original_features_by_category = {k: v.copy() for k, v in features_by_category.items()}

    if uploaded_zip and start_running:
        selectbox_placeholder.empty()
        button_placeholder.empty()

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
                # 修改開始：使用 Faiss 與餘弦相似度計算
                # 準備特徵數據
                category_similarities = {}
                for brand in features_by_category:
                    for category in features_by_category[brand]:
                        labeled_features = features_by_category[brand][category]["labeled_features"]
                        feature_array = np.array([item["features"] for item in labeled_features], dtype=np.float32)
                        # L2 正規化
                        feature_array = l2_normalize(feature_array)
                        # 建立 Faiss 索引（內積）
                        index = faiss.IndexFlatIP(feature_array.shape[1])
                        index.add(feature_array)
                        
                        # 對資料夾中的所有圖像進行查詢
                        folder_similarities = []
                        for img_data in folder_features:
                            img_features = img_data["features"].astype(np.float32).reshape(1, -1)
                            # L2 正規化
                            img_features = l2_normalize(img_features)
                            similarities, _ = index.search(img_features, k=3)
                            avg_similarity = np.mean(similarities)
                            folder_similarities.append(avg_similarity)
                        
                        # 計算該分類的平均相似度
                        avg_similarity = np.mean(folder_similarities)
                        category_similarities[category] = avg_similarity
                
                # 選擇平均相似度最高的分類
                if category_similarities:
                    best_category_name = max(category_similarities, key=category_similarities.get)
                    best_category = {
                        'brand': selected_brand,
                        'category': best_category_name
                    }
                else:
                    st.warning(f"資料夾 {folder} 無法匹配任何分類，跳過此資料夾")
                    continue
                # 修改結束

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
                                # 修改開始：使用 Faiss 查詢特定角度的相似度
                                angle_features = [
                                    item["features"] for item in filtered_by_category 
                                    if item["labels"]["angle"] == angle
                                ]
                                if not angle_features:
                                    continue
                                angle_features = np.array(angle_features, dtype=np.float32)
                                # L2 正規化
                                angle_features = l2_normalize(angle_features)
                                index = faiss.IndexFlatIP(angle_features.shape[1])
                                index.add(angle_features)
                                img_query = l2_normalize(img_features.astype(np.float32).reshape(1, -1))
                                similarities, _ = index.search(img_query, k=1)
                                similarity_percentage = similarities[0][0] * 100
                                # 修改結束
                                valid_angles_by_similarity.append(
                                    (angle, similarity_percentage)
                                )
                            
                            # 根據相似度排序
                            valid_angles_by_similarity.sort(
                                key=lambda x: x[1], reverse=True
                            )
                            
                            for angle, similarity_percentage in valid_angles_by_similarity:
                                if angle not in reassigned_allowed and angle in used_angles:
                                    pass
                                else:
                                    best_angle = angle
                                    best_similarity = similarity_percentage
                                    break
                        
                            if best_angle:
                                used_angles.add(best_angle)  # 標記角度為已使用
                                label_info = {
                                    "資料夾": folder,
                                    "圖片": image_file,
                                    "商品分類": best_category["category"],
                                    "角度": best_angle,
                                    "編號": angle_to_number[best_angle],
                                    "最大相似度": f"{best_similarity:.2f}%"
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

            # 準備特徵數據
            labeled_features = filtered_by_category
            feature_array = np.array([item["features"] for item in labeled_features], dtype=np.float32)
            # L2 正規化
            feature_array = l2_normalize(feature_array)
            labels = [item["labels"] for item in labeled_features]
            # 建立 Faiss 索引（內積）
            index = faiss.IndexFlatIP(feature_array.shape[1])
            index.add(feature_array)

            # 對非特殊圖像進行相似度計算
            for img_data in non_special_images:
                image_file = img_data["image_file"]
                if final_results.get(image_file) is not None:
                    continue

                img_features = img_data["features"].astype(np.float32).reshape(1, -1)
                # L2 正規化
                img_features = l2_normalize(img_features)
                similarities, indices = index.search(img_features, k=len(labels))
                similarities = similarities.flatten()
                # 將相似度轉換為百分比格式（0% 到 100%）
                similarity_percentages = (similarities * 100).clip(0, 100)
                image_similarity_list = []
                for idx, similarity_percentage in zip(indices[0], similarity_percentages):
                    label = labels[idx]
                    item_angle = label["angle"]
                    if is_banned_angle(item_angle, rule_flags):
                        continue
                    image_similarity_list.append({
                        "image_file": image_file,
                        "similarity": similarity_percentage,
                        "label": label,
                        "folder": folder
                    })
                # 去除重複角度
                unique_labels = []
                seen_angles = set()
                for candidate in image_similarity_list:
                    angle = candidate["label"]["angle"]
                    if angle not in seen_angles:
                        unique_labels.append(candidate)
                        seen_angles.add(angle)
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
                                "最大相似度": f"{candidate['similarity']:.2f}%"
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
                            "最大相似度": f"{candidate['similarity']:.2f}%"
                        }
                        used_angles.add(angle)  # 標記角度為已使用
                        assigned_in_this_round.add(image_file)
                    else:
                        max_similarity = -np.inf
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
                            "最大相似度": f"{candidate['similarity']:.2f}%"
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
        for result in results:
            result["圖片"] = os.path.basename(result["圖片"])
        result_df = pd.DataFrame(results)
        st.dataframe(result_df, hide_index=True, use_container_width=True)

        # 創建 '編圖張數與廣告圖' 的資料
        folder_data = []
        for folder in image_folders:
            folder_results = result_df[result_df['資料夾'] == folder]
            valid_images = folder_results[
                (folder_results['編號'] != '超過上限') & (~folder_results['編號'].isna())
            ]
            num_images = len(valid_images)
            # ADS廣告圖
            if selected_brand == "ADS":
                ad_images = valid_images[valid_images['角度'].str.contains('情境|HM')]
                num_ad_images = len(ad_images)
                if num_ad_images > 0:
                    ad_image_value = num_ad_images + 1
                else:
                    ad_image_value = 1
            else:
                # 其他品牌廣告圖欄位
                ad_image_value = ""
                
            folder_data.append({'資料夾': folder, '張數': num_images, '廣告圖': ad_image_value})

        folder_df = pd.DataFrame(folder_data)
        image_type_statistics_df = generate_image_type_statistics(result_df)
        
        # 將結果 DataFrame 和 '編圖張數與廣告圖' 寫入 Excel 檔案
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, sheet_name='編圖結果', index=False)
            folder_df.to_excel(writer, sheet_name='編圖張數與廣告圖', index=False)
            image_type_statistics_df.to_excel(writer, sheet_name='圖片類型統計', index=False)
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
            st.rerun()  
   
#%% 編圖複檢
def get_outer_folder_images(folder_path):
    return sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
    )

def get_prefix(image_files):
    for image_file in image_files:
        filename_without_ext = os.path.splitext(image_file)[0]
        last_underscore_index = filename_without_ext.rfind('_')
        if last_underscore_index != -1:
            return filename_without_ext[:last_underscore_index + 1]
    return ""

def reset_duplicates_flag():
    st.session_state['has_duplicates'] = False

def handle_submission(selected_folder, image_files_to_display, outer_images_to_display, use_full_filename, folder_to_data):
    current_filenames = {}
    # 獲取前綴（僅針對 `1-Main/All`）
    if not use_full_filename:
        prefix = get_prefix(image_files_to_display)
        if prefix == "":
            prefix = get_prefix(image_files)
    else:
        prefix = ""

    # 處理 image_files_to_display 的圖片（僅限 `1-Main/All`）
    for image_file in image_files_to_display:
        text_input_key = f"{selected_folder}_{image_file}"
        new_text = st.session_state.get(text_input_key, "")

        filename_without_ext = os.path.splitext(image_file)[0]
        extension = os.path.splitext(image_file)[1]

        if not use_full_filename:
            last_underscore_index = filename_without_ext.rfind('_')
            if last_underscore_index != -1:
                default_text = filename_without_ext[last_underscore_index + 1:]
            else:
                default_text = filename_without_ext
        else:
            default_text = filename_without_ext + extension

        if new_text.strip() == '':
            new_filename = ''
        else:
            if not use_full_filename:
                new_filename = prefix + new_text + extension
            else:
                new_filename = new_text

        current_filenames[image_file] = {'new_filename': new_filename, 'text': new_text}

    # 處理 outer_images_to_display 的圖片
    for outer_image_file in outer_images_to_display:
        text_input_key = f"outer_{selected_folder}_{outer_image_file}"
        new_text = st.session_state.get(text_input_key, "")

        filename_without_ext = os.path.splitext(outer_image_file)[0]
        extension = os.path.splitext(outer_image_file)[1]

        if not use_full_filename:
            last_underscore_index = filename_without_ext.rfind('_')
            if last_underscore_index != -1:
                default_text = filename_without_ext[last_underscore_index + 1:]
            else:
                default_text = filename_without_ext

            if new_text.strip() == '':
                new_filename = ''
            else:
                new_filename = prefix + new_text + extension
        else:
            default_text = filename_without_ext + extension
            if new_text.strip() == '':
                new_filename = ''
            else:
                new_filename = new_text

        # 移回到 image_files_to_display
        if new_text.strip() != default_text:
            current_filenames[outer_image_file] = {'new_filename': new_filename, 'text': new_text}

    # 檢查重複檔名
    new_filenames = [data['new_filename'] for data in current_filenames.values() if data['new_filename'] != '']
    duplicates = [filename for filename, count in Counter(new_filenames).items() if count > 1]

    if duplicates:
        st.session_state['has_duplicates'] = True
        st.session_state['duplicate_filenames'] = duplicates
        st.session_state['confirmed_changes'][selected_folder] = False
    else:
        st.session_state['has_duplicates'] = False
        st.session_state['confirmed_changes'][selected_folder] = True

        # 僅對 `1-Main/All` 的圖片進行重新命名
        if not use_full_filename:  # 這裡確認是 `1-Main/All` 資料夾
            sorted_files = sorted(current_filenames.items(), key=lambda x: x[1]['new_filename'])
            rename_counter = 1

            for file, data in sorted_files:
                if data['new_filename'] != '':  # 忽略空值檔名
                    new_index = str(rename_counter).zfill(2)  # 01, 02, 03 格式
                    extension = os.path.splitext(file)[1]
                    new_filename = f"{prefix}{new_index}{extension}"

                    # 更新 session state 中的檔名
                    current_filenames[file]['new_filename'] = new_filename
                    current_filenames[file]['text'] = new_index

                    rename_counter += 1

        # 更新 session state 的 filename_changes
        for file, data in current_filenames.items():
            st.session_state['filename_changes'][selected_folder][file] = data

        # 更新 text input 顯示
        for file, data in current_filenames.items():
            text_input_key = f"{selected_folder}_{file}"
            st.session_state[text_input_key] = data['text']

    # 自動調整張數值
    num_outer_images = len([file for file, data in current_filenames.items() if data['new_filename'] == ''])
    if folder_to_data:
        num_images_key = f"{selected_folder}_num_images"
        if num_images_key in st.session_state:
            current_num_images = int(st.session_state[num_images_key])
            st.session_state[num_images_key] = str(max(1, current_num_images - num_outer_images))

        ad_images_key = f"{selected_folder}_ad_images"
        ad_images_value = st.session_state.get(ad_images_key)
        data = folder_to_data.get(selected_folder, {})
        data_folder_name = data.get('資料夾')
        if data_folder_name:
            st.session_state['folder_values'][data_folder_name] = {
                '張數': st.session_state[num_images_key],
                '廣告圖': ad_images_value
            }

def get_sort_key(image_file):
    if selected_folder in st.session_state['filename_changes'] and image_file in st.session_state['filename_changes'][selected_folder]:
        data = st.session_state['filename_changes'][selected_folder][image_file]
        new_filename = data['new_filename']
        if new_filename != '':
            return new_filename
        else:
            # 使用最近非空檔名排序
            return data.get('last_non_empty', image_file)
    else:
        return image_file
    
with tab2:
    initialize_session_state()
    
    st.write("\n")

    uploaded_file = st.file_uploader(
        "上傳編圖結果 Zip 檔",
        type=["zip"],
        key='file_uploader_' + str(st.session_state['file_uploader_key2'])
    )

    if uploaded_file is None:
        st.session_state['filename_changes'] = {}
        st.session_state['confirmed_changes'] = {}
        st.session_state['image_cache'] = {}
        st.session_state['folder_values'] = {}

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            with zipfile.ZipFile(uploaded_file) as zip_ref:
                zip_ref.extractall(tmpdirname)

            # 讀取 '編圖結果.xlsx' 並構建資料夾對應關係
            excel_file_path = os.path.join(tmpdirname, '編圖結果.xlsx')
            if os.path.exists(excel_file_path):
                excel_sheets = pd.read_excel(excel_file_path, sheet_name=None)
                if '編圖張數與廣告圖' in excel_sheets:
                    sheet_df = excel_sheets['編圖張數與廣告圖']
                    folder_to_row_idx = {}
                    for idx, row in sheet_df.iterrows():
                        folder_name = str(row['資料夾'])
                        folder_to_row_idx[folder_name] = idx
                else:
                    sheet_df = None
                    folder_to_row_idx = {}
            else:
                excel_sheets = {}
                sheet_df = None
                folder_to_row_idx = {}

            # 建立資料夾名稱與 '資料夾' 值的對應關係
            folder_to_data = {}
            top_level_folders = [name for name in os.listdir(tmpdirname) if os.path.isdir(os.path.join(tmpdirname, name))]

            for folder_name in top_level_folders:
                matched = False
                for data_folder_name in folder_to_row_idx.keys():
                    if data_folder_name in folder_name:
                        idx = folder_to_row_idx[data_folder_name]
                        row = sheet_df.loc[idx]
                        folder_to_data[folder_name] = {
                            '資料夾': data_folder_name,
                            '張數': str(row['張數']),
                            '廣告圖': str(row['廣告圖'])
                        }
                        matched = True
                        break
                if not matched:
                    folder_to_data[folder_name] = {
                        '資料夾': None,
                        '張數': '1',
                        '廣告圖': '1'
                    }

            if 'previous_selected_folder' not in st.session_state and top_level_folders:
                st.session_state['previous_selected_folder'] = top_level_folders[0]

            if top_level_folders:
                if 'previous_selected_folder' not in st.session_state:
                    st.session_state['previous_selected_folder'] = None
                
                if 'last_text_inputs' not in st.session_state:
                    st.session_state['last_text_inputs'] = {}
                
                previous_folder = st.session_state['previous_selected_folder']
                selected_folder = st.pills(
                    "選擇一個資料夾",
                    top_level_folders,
                    default=top_level_folders[0],
                    label_visibility="collapsed",
                    on_change=reset_duplicates_flag
                )
                
                # 當 `selected_folder` 變成 `None` 時，保存目前的 text_input 值
                if selected_folder is None and previous_folder is not None:
                    st.session_state['last_text_inputs'][previous_folder] = {
                        key: st.session_state[key]
                        for key in st.session_state if key.startswith(f"{previous_folder}_")
                    }
                
                # 從 `None` 切回之前的資料夾時，恢復 text_input 值
                if selected_folder is not None and previous_folder is None:
                    if selected_folder in st.session_state['last_text_inputs']:
                        for key, value in st.session_state['last_text_inputs'][selected_folder].items():
                            st.session_state[key] = value
                
                # 更新 previous_selected_folder
                st.session_state['previous_selected_folder'] = selected_folder

                st.write("\n")

                if selected_folder is None:
                    st.stop()

                img_folder_path = os.path.join(tmpdirname, selected_folder, '2-IMG')
                use_full_filename = False
                if not os.path.exists(img_folder_path):
                    img_folder_path = os.path.join(tmpdirname, selected_folder, '1-Main', 'All')
                    use_full_filename = False
                else:
                    use_full_filename = True

                # 檢查最外層資料夾圖片
                outer_folder_path = os.path.join(tmpdirname, selected_folder)

                outer_images = get_outer_folder_images(outer_folder_path)

                if os.path.exists(img_folder_path):
                    image_files = get_outer_folder_images(img_folder_path)

                    if image_files:
                        if selected_folder not in st.session_state['filename_changes']:
                            st.session_state['filename_changes'][selected_folder] = {}
                        if selected_folder not in st.session_state['confirmed_changes']:
                            st.session_state['confirmed_changes'][selected_folder] = False
                        if selected_folder not in st.session_state['image_cache']:
                            st.session_state['image_cache'][selected_folder] = {}

                        # 重建 image_files_to_display 和 outer_images_to_display_updated
                        all_images = set(image_files + outer_images)

                        image_files_to_display = []
                        outer_images_to_display_updated = []

                        for image_file in all_images:
                            if selected_folder in st.session_state['filename_changes'] and image_file in st.session_state['filename_changes'][selected_folder]:
                                data = st.session_state['filename_changes'][selected_folder][image_file]
                                if data['new_filename'] == '':
                                    outer_images_to_display_updated.append(image_file)
                                else:
                                    image_files_to_display.append(image_file)
                            else:
                                if image_file in image_files:
                                    image_files_to_display.append(image_file)
                                else:
                                    outer_images_to_display_updated.append(image_file)

                        image_files_to_display.sort(key=get_sort_key)
                        outer_images_to_display_updated.sort(key=get_sort_key)

                        with st.form(f"filename_form_{selected_folder}"):
                            cols = st.columns(6)
                            for idx, image_file in enumerate(image_files_to_display):
                                if idx % 6 == 0 and idx != 0:
                                    cols = st.columns(6)
                                col = cols[idx % 6]

                                # 總是使用原始檔名來讀取圖片
                                image_path = os.path.join(img_folder_path, image_file) if image_file in image_files else os.path.join(outer_folder_path, image_file)
                                if image_path not in st.session_state['image_cache'][selected_folder]:
                                    image = Image.open(image_path)
                                    image = ImageOps.pad(image, (800, 800), method=Image.Resampling.LANCZOS)
                                    st.session_state['image_cache'][selected_folder][image_path] = image
                                else:
                                    image = st.session_state['image_cache'][selected_folder][image_path]

                                col.image(image, use_container_width=True)

                                filename_without_ext = os.path.splitext(image_file)[0]
                                extension = os.path.splitext(image_file)[1]

                                if use_full_filename:
                                    default_text = filename_without_ext + extension
                                else:
                                    last_underscore_index = filename_without_ext.rfind('_')
                                    if last_underscore_index != -1:
                                        default_text = filename_without_ext[last_underscore_index + 1:]
                                    else:
                                        default_text = filename_without_ext

                                if (selected_folder in st.session_state['filename_changes'] and
                                    image_file in st.session_state['filename_changes'][selected_folder]):
                                    modified_text = st.session_state['filename_changes'][selected_folder][image_file]['text']
                                else:
                                    modified_text = default_text

                                text_input_key = f"{selected_folder}_{image_file}"
                                # 初始化 session state，如果 key 不存在則賦予 modified_text
                                if text_input_key not in st.session_state:
                                    st.session_state[text_input_key] = modified_text
                                
                                # 使用 session state 的值建立 text_input
                                col.text_input('檔名', key=text_input_key, label_visibility="collapsed")


                            # 顯示最外層資料夾圖片的 popover
                            col1, col2, col3 ,col4= st.columns([1.1,1.71,1.23, 1.23], vertical_alignment="center")
                            if outer_images_to_display_updated:
                                with col4.popover("查看外層圖片"):
                                    outer_cols = st.columns(6)
                                    for idx, outer_image_file in enumerate(outer_images_to_display_updated):
                                        if idx % 6 == 0 and idx != 0:
                                            outer_cols = st.columns(6)
                                        col = outer_cols[idx % 6]

                                        # 總是使用原始檔名來讀取圖片
                                        outer_image_path = os.path.join(outer_folder_path, outer_image_file) if outer_image_file in outer_images else os.path.join(img_folder_path, outer_image_file)

                                        if outer_image_path not in st.session_state['image_cache'][selected_folder]:
                                            outer_image = Image.open(outer_image_path)
                                            outer_image = ImageOps.pad(outer_image, (800, 800), method=Image.Resampling.LANCZOS)
                                            st.session_state['image_cache'][selected_folder][outer_image_path] = outer_image
                                        else:
                                            outer_image = st.session_state['image_cache'][selected_folder][outer_image_path]

                                        col.image(outer_image, use_container_width=True)
                                        filename_without_ext = os.path.splitext(outer_image_file)[0]
                                        extension = os.path.splitext(outer_image_file)[1]

                                        if use_full_filename:
                                            default_text = filename_without_ext + extension
                                        else:
                                            last_underscore_index = filename_without_ext.rfind('_')
                                            if last_underscore_index != -1:
                                                default_text = filename_without_ext[last_underscore_index + 1:]
                                            else:
                                                default_text = filename_without_ext

                                        if (selected_folder in st.session_state['filename_changes'] and
                                            outer_image_file in st.session_state['filename_changes'][selected_folder]):
                                            modified_text = st.session_state['filename_changes'][selected_folder][outer_image_file]['text']
                                            if modified_text == '':
                                                # 顯示最近非空檔名
                                                modified_text = st.session_state['filename_changes'][selected_folder][outer_image_file].get('last_non_empty', default_text)
                                        else:
                                            modified_text = default_text

                                        text_input_key = f"outer_{selected_folder}_{outer_image_file}"
                                        col.text_input('檔名', value=modified_text, key=text_input_key)

                            if folder_to_data and folder_to_data.get(selected_folder, {}).get('資料夾'):
                                # 新增張數和廣告圖的選擇框
                                data = folder_to_data.get(selected_folder, {})
                                data_folder_name = data.get('資料夾')
                                if data_folder_name and 'folder_values' in st.session_state and data_folder_name in st.session_state['folder_values']:
                                    num_images_default = st.session_state['folder_values'][data_folder_name]['張數']
                                    ad_images_default = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                else:
                                    num_images_default = data.get('張數', '1')
                                    ad_images_default = data.get('廣告圖', '1')

                                num_images_key = f"{selected_folder}_num_images"
                                ad_images_key = f"{selected_folder}_ad_images"
                                num_images_options = [str(i) for i in range(1, 11)]
                                ad_images_options = [str(i) for i in range(1, 11)]
                                if outer_images_to_display_updated:
                                    with col3.popover("編圖數/廣告圖"):
                                        colA,colB = st.columns(2)
                                        colA.selectbox('張數', num_images_options, index=num_images_options.index(num_images_default), key=num_images_key)
                                        colB.selectbox('廣告圖', ad_images_options, index=ad_images_options.index(ad_images_default), key=ad_images_key)
                                        st.warning('若有修改記得點擊 "確認修改"')
                                else:
                                    with col4.popover("編圖數/廣告圖"):
                                        colA,colB = st.columns(2)
                                        colA.selectbox('張數', num_images_options, index=num_images_options.index(num_images_default), key=num_images_key)
                                        colB.selectbox('廣告圖', ad_images_options, index=ad_images_options.index(ad_images_default), key=ad_images_key)
                                        st.warning('若有修改記得點擊 "確認修改"')
                            else:
                                # 如果沒有編圖結果.xlsx，則不顯示 selectbox
                                num_images_key = None
                                ad_images_key = None
                                folder_to_data = None

                            col1.form_submit_button(
                                "確認修改",
                                on_click=handle_submission,
                                args=(selected_folder, image_files_to_display, outer_images_to_display_updated, use_full_filename, folder_to_data )
                                )
                            if st.session_state.get('has_duplicates') == True:
                                col2.warning(f"檔名重複: {', '.join(st.session_state['duplicate_filenames'])}")

                        if any(st.session_state['confirmed_changes'].values()):
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                                # 找出頂層的非資料夾檔案
                                top_level_files = [name for name in os.listdir(tmpdirname) if os.path.isfile(os.path.join(tmpdirname, name))]

                                # 先將頂層的非資料夾檔案加入 zip
                                for file_name in top_level_files:
                                    file_path = os.path.join(tmpdirname, file_name)
                                    arcname = file_name
                                    # 如果是編圖結果.xlsx，且有需要更新
                                    if file_name == '編圖結果.xlsx' and excel_sheets and folder_to_data:
                                        # 更新 '張數' 和 '廣告圖' 欄位
                                        df = excel_sheets['編圖張數與廣告圖']
                                        for idx, row in df.iterrows():
                                            data_folder_name = str(row['資料夾'])
                                            if data_folder_name in st.session_state['folder_values']:
                                                df.at[idx, '張數'] = int(st.session_state['folder_values'][data_folder_name]['張數'])
                                                df.at[idx, '廣告圖'] = int(st.session_state['folder_values'][data_folder_name]['廣告圖'])

                                        # 重寫 Excel 文件
                                        output = BytesIO()
                                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                            for sheet_name, df_sheet in excel_sheets.items():
                                                if sheet_name == '編圖張數與廣告圖':
                                                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                                                else:
                                                    df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                                        output.seek(0)
                                        zipf.writestr('編圖結果.xlsx', output.getvalue())
                                    elif file_name == '編圖結果.xlsx' and not folder_to_data:
                                        # 如果沒有編圖結果.xlsx，不包含在輸出 ZIP 中
                                        continue
                                    else:
                                        zipf.write(file_path, arcname=arcname)

                                # 再處理各個資料夾中的檔案
                                for folder_name in top_level_folders:
                                    folder_path = os.path.join(tmpdirname, folder_name)
                                    for root, dirs, files in os.walk(folder_path):
                                        for file in files:
                                            full_path = os.path.join(root, file)
                                            rel_path = os.path.relpath(full_path, tmpdirname)
                                            path_parts = rel_path.split(os.sep)

                                            original_file = file  # 保留原始檔名
                                            if folder_name in st.session_state['filename_changes'] and original_file in st.session_state['filename_changes'][folder_name]:
                                                data = st.session_state['filename_changes'][folder_name][original_file]
                                                new_filename = data['new_filename']
                                                if new_filename.strip() == '':
                                                    # 檔名為空，放在最外層資料夾
                                                    new_rel_path = os.path.join(folder_name, original_file)
                                                    zipf.write(full_path, arcname=new_rel_path)
                                                else:
                                                    # 根據 use_full_filename 放置到正確的資料夾
                                                    if use_full_filename:
                                                        # 放在 2-IMG
                                                        idx = path_parts.index(folder_name)
                                                        path_parts = path_parts[:idx+1] + ['2-IMG', new_filename]
                                                    else:
                                                        # 放在 1-Main/All
                                                        idx = path_parts.index(folder_name)
                                                        path_parts = path_parts[:idx+1] + ['1-Main', 'All', new_filename]
                                                    new_rel_path = os.path.join(*path_parts)
                                                    zipf.write(full_path, arcname=new_rel_path)
                                            else:
                                                zipf.write(full_path, arcname=rel_path)

                            zip_buffer.seek(0)
                            st.write("\n")
                            st.download_button(
                                label='下載修改後的檔案',
                                data=zip_buffer,
                                file_name=uploaded_file.name,
                                mime='application/zip',
                                on_click=reset_file_uploader
                            )
                    else:
                        st.error("未找到圖片。")
                else:
                    st.error("不存在 '2-IMG' 或 '1-Main/All' 資料夾。")
            else:
                st.error("未找到任何資料夾。")
