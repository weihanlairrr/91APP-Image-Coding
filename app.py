#%% 導入區
import streamlit as st
import pandas as pd
import zipfile
import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageOps, ImageDraw, ImageFont
from io import BytesIO
import pickle
import shutil
import numpy as np
import re
import tempfile
from collections import Counter, defaultdict
import chardet
import faiss  
import functools

st.set_page_config(page_title='TP自動化編圖工具', page_icon='👕')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
faiss.omp_set_num_threads(1)

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

def get_dynamic_nlist(num_samples):
    """
    根據資料數量動態決定 nlist。
    參數:
        num_samples: 資料數量
    回傳:
        適合的 nlist 值
    """
    if num_samples >= 1000:
        return min(200, int(np.sqrt(num_samples)))  # 大量資料，使用較高 nlist
    elif num_samples >= 100:
        return min(100, int(np.sqrt(num_samples)))  # 中等資料，使用中等 nlist
    else:
        return max(1, num_samples // 2)  # 少量資料，降低 nlist

@st.cache_resource
def load_image_features_with_ivf(train_file_path):
    """
    載入商品特徵並構建 Faiss 倒排索引。
    參數:
        train_file_path: .pkl 檔案的路徑
    回傳:
        包含倒排索引和其他特徵信息的字典
    """
    with open(train_file_path, 'rb') as f:
        features_by_category = pickle.load(f)
    
    # 為每個分類構建倒排索引
    for brand, categories in features_by_category.items():
        for category, data in categories.items():
            features = np.array([item["features"] for item in data["labeled_features"]], dtype=np.float32)
            features = l2_normalize(features)  # L2 正規化
            num_samples = len(features)
            nlist = get_dynamic_nlist(num_samples)  # 動態計算 nlist
            index = build_ivf_index(features, nlist)
            features_by_category[brand][category]["index"] = index
    return features_by_category

def build_ivf_index(features, nlist):
    """
    使用倒排索引構建 Faiss 索引。
    參數:
        features: numpy array，形狀為 (n_samples, n_features)
        nlist: 分簇數量
    回傳:
        Faiss 索引
    """
    d = features.shape[1]  # 特徵向量的維度
    nlist = min(nlist, len(features))  # 確保簇數量不超過樣本數
    quantizer = faiss.IndexFlatIP(d)  # 用於分簇的基礎索引，使用內積
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(features)  # 訓練索引，生成簇心
    index.add(features)  # 添加數據到索引
    return index

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
    
def handle_file_uploader_change():
    """當 file uploader 狀態變化時更新 text area disabled 屬性"""
    file_key = 'file_uploader_' + str(st.session_state.get('file_uploader_key1', 0))
    uploaded_file = st.session_state.get(file_key, None)
    st.session_state.text_area_disabled = bool(uploaded_file)

def handle_text_area_change():
    """當 text area 狀態變化時更新 file uploader disabled 屬性"""
    text_key = 'text_area_' + str(st.session_state.get('text_area_key1', 0))
    text_content = st.session_state.get(text_key, "")
    
    # 如果輸入的內容是 search-ms 開頭的 URI，進行路徑轉換
    if text_content.startswith("search-ms:"):
        # 解析路徑中的 location 欄位
        match = re.search(r'location:([^&]+)', text_content)
        if match:
            # 解碼 URL 並轉換為實際路徑
            decoded_path = re.sub(r'%3A', ':', match.group(1))  # 解碼冒號
            decoded_path = re.sub(r'%5C', '\\\\', decoded_path)  # 解碼反斜線
            st.session_state[text_key] = decoded_path
        else:
            st.warning("無法解析 search-ms 路徑，請確認輸入格式。")
    
    # 更新 file uploader 的狀態
    st.session_state.file_uploader_disabled = bool(st.session_state[text_key])
            
def reset_key_tab1():
    """
    重置文件上傳器的狀態，並刪除上傳的圖像和臨時壓縮檔。
    """
    st.session_state['file_uploader_key1'] += 1 
    st.session_state['text_area_key1'] += 1 
    st.session_state['file_uploader_disabled'] = False
    st.session_state['text_area_disabled'] = False

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
    獲取指定資料夾中的所有圖像檔案，並判斷是否使用 2-IMG 的邏輯。
    參數:
        folder_path: 資料夾的路徑
    回傳:
        圖像檔案的相對路徑和完整路徑的列表，以及是否使用 2-IMG 的邏輯
    """
    image_files = []
    two_img_folder_path = os.path.join(folder_path, '2-IMG')
    ads_folder_path = os.path.join(folder_path, '1-Main/All')
    use_two_img_folder = False

    if os.path.exists(two_img_folder_path) and os.path.isdir(two_img_folder_path):
        use_two_img_folder = True
        # 只處理 '2-IMG' 資料夾內的圖片
        for file in os.listdir(two_img_folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                full_image_path = os.path.join(two_img_folder_path, file)
                relative_image_path = os.path.relpath(full_image_path, folder_path)
                image_files.append((relative_image_path, full_image_path))
    elif os.path.exists(ads_folder_path) and os.path.isdir(ads_folder_path):
        for file in os.listdir(ads_folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                full_image_path = os.path.join(ads_folder_path, file)
                relative_image_path = os.path.relpath(full_image_path, folder_path)
                image_files.append((relative_image_path, full_image_path))
    else:
        # 原本的邏輯，遍歷整個資料夾
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
    return image_files, use_two_img_folder

def rename_numbers_in_folder(results, folder_label_limits, folder_start_numbers, angle_to_prefix, category_label_limits, category_start_numbers):
    """
    根據編號重新命名資料夾中的圖像檔案，並根據前綴獨立編號。
    參數:
        results: 圖像處理的結果列表
        folder_label_limits: 每個前綴的主圖上限字典
        folder_start_numbers: 每個前綴的起始編號字典
        angle_to_prefix: 角度與指定前綴的對應字典
        category_label_limits: 商品分類的主圖上限字典
        category_start_numbers: 商品分類的起始編號字典
    回傳:
        更新後的結果列表
    """
    folders = set([result["資料夾"] for result in results])  # 獲取所有資料夾名稱
    for folder in folders:
        folder_results = [r for r in results if r["資料夾"] == folder]
        # 針對不同的前綴進行分組
        prefix_groups = defaultdict(list)
        for result in folder_results:
            angle = result["角度"]
            category = result["商品分類"]
            # 檢查指定前綴的適用性
            specified_prefix = angle_to_prefix.get((angle, category))
            if specified_prefix is None:
                specified_prefix = angle_to_prefix.get((angle, None))
            prefix = specified_prefix if specified_prefix else folder
            result["前綴"] = prefix  # 暫時添加前綴到結果中，方便處理
            prefix_groups[prefix].append(result)
        for prefix, prefix_results in prefix_groups.items():
            # 檢查是否有未編號的圖像
            if any(pd.isna(r["編號"]) or r["編號"] == "" for r in prefix_results):
                continue
            # 按照編號排序
            prefix_results.sort(key=lambda x: int(x["編號"]))
            # 獲取該前綴對應的商品分類
            category = None
            for r in prefix_results:
                if r["商品分類"]:
                    category = r["商品分類"]
                    break
            # 根據商品分類獲取 label_limit 和 starting_number
            label_limit = int(category_label_limits.get(category, other_label_limit))
            starting_number = int(category_start_numbers.get(category, other_start_number))
            for idx, result in enumerate(prefix_results):
                if idx < label_limit:
                    result["編號"] = f'{starting_number + idx:02}'  # 根據起始編號進行編碼
                else:
                    result["編號"] = "超過上限"  # 超過編號上限時標記
            # 移除臨時添加的前綴欄位
            for result in prefix_results:
                del result["前綴"]
    return results

def rename_and_zip_folders(results, output_excel_data, skipped_images, folder_settings, angle_to_prefix):
    """
    重新命名圖像檔案並壓縮處理後的資料夾和結果 Excel 檔。
    參數:
        results: 圖像處理的結果列表
        output_excel_data: 結果的 Excel 資料
        skipped_images: 被跳過的圖像列表
        folder_settings: 每個資料夾是否使用 2-IMG 的邏輯
        angle_to_prefix: 角度與指定前綴的對應字典
    回傳:
        壓縮檔的二進位數據
    """
    output_folder_path = "uploaded_images"  # 根資料夾

    for result in results:
        folder_name = result["資料夾"]
        image_file = result["圖片"]
        new_number = result["編號"]
        angle = result["角度"]  # 獲取分配的角度
        category = result["商品分類"]  # 獲取商品分類

        # 檢查指定前綴的適用性
        specified_prefix = angle_to_prefix.get((angle, category))
        if specified_prefix is None:
            specified_prefix = angle_to_prefix.get((angle, None))

        prefix = specified_prefix if specified_prefix else folder_name

        # 設定主資料夾路徑
        folder_path = os.path.join(output_folder_path, folder_name)
        use_two_img_folder = folder_settings.get(folder_name, False)  # 根據該資料夾的設置

        if use_two_img_folder:
            main_folder_structure = "2-IMG"
        else:
            main_folder_structure = "1-Main/All"
        main_folder_path = os.path.join(folder_path, main_folder_structure)
        os.makedirs(main_folder_path, exist_ok=True)  # 創建主資料夾

        old_image_path = os.path.join(folder_path, image_file)
        file_extension = os.path.splitext(image_file)[1]  # 取得原始檔案的副檔名

        if (
            use_two_img_folder
            and (new_number == "超過上限" or any(keyword in image_file for keyword in keywords_to_skip))
        ):
            new_image_path = old_image_path
        elif new_number == "超過上限" or pd.isna(new_number):
            new_image_path = os.path.join(folder_path, os.path.basename(image_file))
        else:
            if use_two_img_folder:
                new_image_name = f"{prefix}{new_number}{file_extension}"  # 2-IMG 圖片移除底線
            else:
                new_image_name = f"{prefix}_{new_number}{file_extension}"  # 其他情況保留底線

            new_image_path = os.path.join(main_folder_path, new_image_name)

        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

        if os.path.exists(old_image_path) and old_image_path != new_image_path:
            os.rename(old_image_path, new_image_path)  # 重新命名或移動圖像檔案

    # 處理 skipped_images，將其移動到最外層資料夾（僅當非 2-IMG）
    for skipped in skipped_images:
        folder_name = skipped["資料夾"]
        image_file = skipped["圖片"]

        folder_path = os.path.join(output_folder_path, folder_name)
        old_image_path = os.path.join(folder_path, image_file)

        use_two_img_folder = folder_settings.get(folder_name, False)

        if not use_two_img_folder:
            new_image_path = os.path.join(folder_path, os.path.basename(image_file))  # 保留在最外層資料夾

            if os.path.exists(old_image_path):
                os.rename(old_image_path, new_image_path)  # 移動到最外層

    zip_buffer = BytesIO()  # 創建內存中的緩衝區
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder in os.listdir("uploaded_images"):
            folder_path = os.path.join("uploaded_images", folder)
            if os.path.isdir(folder_path):
                use_two_img_folder = folder_settings.get(folder, False)
                if use_two_img_folder:
                    new_folder_name = folder  # 不添加 "_OK"
                else:
                    new_folder_name = f"{folder}_OK"  # 添加 "_OK"

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
    # ---- 修改開始：僅計算實際有更改檔名的圖片 ----
    filtered_results = results[(results["編號"] != "超過上限") & (~results["編號"].isna())]
    # ---- 修改結束 ----
    for folder, folder_results in filtered_results.groupby("資料夾"):
        # 統計含有"模特"的角度數量
        model_count = folder_results["角度"].str.contains("模特").sum()
        
        # 統計符合"平拍"的角度，排除 HM1-HM10
        excluded_angles = {"HM1", "HM2", "HM3", "HM4", "HM5", "HM6", "HM7", "HM8", "HM9", "HM10"}
        flat_lay_count = folder_results["角度"].apply(
            lambda x: x not in excluded_angles and "模特" not in x 
        ).sum()
        
        # 儲存資料夾的統計結果
        statistics.append({
            "資料夾": folder,
            "模特": model_count,
            "平拍": flat_lay_count,
        })
    
    return pd.DataFrame(statistics)

#%% 自動編圖介面
tab1, tab2, tab3 = st.tabs(["自動編圖", "編圖複檢", "覆蓋舊檔案與刪外層圖"])
with tab1:
    resnet = load_resnet_model()
    preprocess = get_preprocess_transforms()

    brand_dependencies = {
        "ADS": {
            "train_file": "dependencies/image_features.pkl",
            "angle_filename_reference": "dependencies/ADS檔名角度對照表.xlsx",
        },
    }

    # 初始化 session state
    if 'file_uploader_key1' not in st.session_state:
        st.session_state['file_uploader_key1'] = 0
    if 'text_area_key1' not in st.session_state:
        st.session_state['text_area_key1'] = 0
    if 'file_uploader_disabled' not in st.session_state:
        st.session_state['file_uploader_disabled'] = False
    if 'text_area_disabled' not in st.session_state:
        st.session_state['text_area_disabled'] = False
    if 'text_area_content' not in st.session_state:
        st.session_state['text_area_content'] = ""

    brand_list = list(brand_dependencies.keys())
    st.write("\n")
    col1, col2 = st.columns([1.6, 1])

    # 檔案上傳元件
    uploaded_zip = col1.file_uploader(
        "上傳 ZIP 檔案",
        type=["zip"],
        key='file_uploader_' + str(st.session_state['file_uploader_key1']),
        disabled=st.session_state['file_uploader_disabled'],
        on_change=handle_file_uploader_change
    )

    # 資料夾路徑輸入元件
    input_path = col2.text_area(
        "或 輸入資料夾路徑",
        height=78,
        key='text_area_' + str(st.session_state['text_area_key1']),
        disabled=st.session_state['text_area_disabled'],
        on_change=handle_text_area_change
    )

    start_running = False
    if input_path:
        st.session_state["input_path_from_tab1"] = input_path  # 儲存路徑

    if uploaded_zip or input_path:
        col1, col2, col3 = st.columns([1.5, 2, 2], vertical_alignment="center", gap="medium")
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
        
        # 讀取基本設定
        category_settings_df = pd.read_excel(angle_filename_reference, sheet_name="基本設定")
        category_label_limits = dict(zip(category_settings_df.iloc[:, 0], category_settings_df.iloc[:, 1]))
        category_start_numbers = dict(zip(category_settings_df.iloc[:, 0], category_settings_df.iloc[:, 2]))
        
        # 讀取 "其他" 的設定值
        other_label_limit = category_label_limits.get("其他", 10)
        other_start_number = category_start_numbers.get("其他", 1)
        
        keywords_to_skip = pd.read_excel(angle_filename_reference, sheet_name='不編的檔名', usecols=[0]).iloc[:, 0].dropna().astype(str).tolist()
        substitute_df = pd.read_excel(angle_filename_reference, sheet_name='有條件使用的檔名', usecols=[0, 1])
        substitute = [{"set_a": row.iloc[0].split(','), "set_b": row.iloc[1].split(',')} for _, row in substitute_df.iterrows()]
        
        reassigned_allowed = pd.read_excel(angle_filename_reference, sheet_name='可以重複分配的角度', usecols=[0]).iloc[:, 0].dropna().tolist()
        
        angle_banning_df = pd.read_excel(angle_filename_reference, sheet_name='角度禁止規則', usecols=[0, 1, 2])
        angle_banning_rules = [{"if_appears_in_angle": row.iloc[0].split(','), "banned_angle": row.iloc[1], "banned_angle_logic": row.iloc[2]} for _, row in angle_banning_df.iterrows()]
        
        category_rules_df = pd.read_excel(angle_filename_reference, sheet_name='商品分類及關鍵字條件', usecols=[0, 1, 2])
        category_rules = {row.iloc[0]: {"keywords": row.iloc[1].split(','), "match_all": row.iloc[2]} for _, row in category_rules_df.iterrows()}
        
        features_by_category = load_image_features_with_ivf(train_file)
        # 複製一份原始特徵資料，避免在處理時修改到原始資料
        original_features_by_category = {k: v.copy() for k, v in features_by_category.items()}

    if (uploaded_zip or input_path) and start_running:
        selectbox_placeholder.empty()
        button_placeholder.empty()

        if os.path.exists("uploaded_images"):
            shutil.rmtree("uploaded_images")
            
        if uploaded_zip:
            # 將上傳的 zip 檔案寫入臨時檔案
            with open("temp.zip", "wb") as f:
                f.write(uploaded_zip.getbuffer())
        
            # 解壓上傳的 zip 檔案
            unzip_file("temp.zip")
        elif input_path:
            # 複製 input_path 下的所有資料夾到 'uploaded_images' 資料夾
            if not os.path.exists(input_path):
                st.error("指定的本地路徑不存在，請重新輸入。")
                st.stop()
            else:
                shutil.copytree(input_path, "uploaded_images")

        # 初始化特殊映射字典
        special_mappings = {}
        # 建立角度與指定前綴的對應關係
        angle_to_prefix = {}
        prefix_to_category = {}  # 初始化 prefix_to_category

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
                        category_filename = [x.strip() for x in category_filename_raw.split(',')]  # 支持多個條件
                    else:
                        category = category_raw
                        category_filename = None
                angle = str(row.iloc[2]).strip()
                angles = [a.strip() for a in angle.split(',')]
                # 處理指定前綴
                prefix = row.iloc[3] if len(row) > 3 and not pd.isna(row.iloc[3]) else None

                special_mappings[keyword] = {
                    'category': category, 
                    'category_filename': category_filename,
                    'angles': angles,
                    'prefix': prefix  # 儲存指定前綴
                }

                # 建立角度與指定前綴的對應
                for a in angles:
                    angle_to_prefix[(a, category)] = prefix  # 修改為以 (angle, category) 作為鍵

                # 新增此行：建立 prefix_to_category 映射
                if prefix:
                    prefix_to_category[prefix] = category
        
        # 新增：建立 folder_label_limits 和 folder_start_numbers
        folder_label_limits = {}
        folder_start_numbers = {}

        for prefix, category in prefix_to_category.items():
            label_limit = category_label_limits.get(category, other_label_limit)
            starting_number = category_start_numbers.get(category, other_start_number)
            folder_label_limits[prefix] = int(label_limit)
            folder_start_numbers[prefix] = int(starting_number)

        # 以下新增：處理沒有指定前綴的情況，使用對應的商品分類設定
        # 如果有資料夾名稱作為前綴，且未在 folder_label_limits 中，則嘗試從商品分類獲取設定
        all_prefixes = set(prefix_to_category.keys())
        for folder in os.listdir("uploaded_images"):
            if folder not in all_prefixes:
                # 嘗試從商品分類中獲取設定
                label_limit = category_label_limits.get(folder, other_label_limit)
                starting_number = category_start_numbers.get(folder, other_start_number)
                folder_label_limits[folder] = int(label_limit)
                folder_start_numbers[folder] = int(starting_number)

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
    
        folder_settings = {}  # 存儲每個資料夾是否使用 2-IMG 的邏輯

        # 遍歷每個圖像資料夾進行處理
        for folder in image_folders:
            # 每次處理新資料夾前，重置 features_by_category
            features_by_category = {k: v.copy() for k, v in original_features_by_category.items()}
    
            folder_path = os.path.join("uploaded_images", folder)
            image_files, use_two_img_folder = get_images_in_folder(folder_path)  # 獲取資料夾中的圖像檔案
            folder_settings[folder] = use_two_img_folder  # 存儲該資料夾的設定

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
                                # 支持多個 category_filename 條件
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
                # 修改開始：使用 IndexIVFFlat 與餘弦相似度計算
                # 準備特徵數據
                category_similarities = {}
                for brand in features_by_category:
                    for category in features_by_category[brand]:
                        index = features_by_category[brand][category]["index"]  # 使用預先構建的索引
                        num_samples = len(features_by_category[brand][category]["labeled_features"])
                        nlist = index.nlist
                        nprobe = max(1, int(np.sqrt(nlist)))
                        index.nprobe = nprobe  # 設定搜尋的簇數量
                        folder_similarities = []
                        
                        for img_data in folder_features:
                            img_features = img_data["features"].astype(np.float32).reshape(1, -1)
                            img_features = l2_normalize(img_features)
                            similarities, _ = index.search(img_features, k=5)
                            avg_similarity = np.mean(similarities)
                            folder_similarities.append(avg_similarity)
                        
                        category_similarities[category] = np.mean(folder_similarities)
                
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

            # 根據分類設定 label_limit 和 starting_number
            if best_category["category"] in category_label_limits:
                label_limit = int(category_label_limits[best_category["category"]])
            else:
                label_limit = int(other_label_limit)

            if best_category["category"] in category_start_numbers:
                starting_number = int(category_start_numbers[best_category["category"]])
            else:
                starting_number = int(other_start_number)

            # 以下新增：為每個資料夾的前綴設定 label_limit 和 starting_number
            # 取得此資料夾的前綴
            folder_prefixes = set()
            for img_data in folder_features:
                image_file = img_data["image_file"]
                special_angles = img_data["special_angles"]
                special_category = img_data["special_category"]
                # 獲取前綴
                angle = None
                category = best_category["category"]
                if special_angles:
                    angle = special_angles[0]
                    category = special_category if special_category else category
                specified_prefix = angle_to_prefix.get((angle, category))
                if specified_prefix is None:
                    specified_prefix = angle_to_prefix.get((angle, None))
                prefix = specified_prefix if specified_prefix else folder
                folder_prefixes.add(prefix)
            # 為此資料夾的所有前綴設定 label_limit 和 starting_number
            for prefix in folder_prefixes:
                if prefix not in folder_label_limits:
                    label_limit = int(category_label_limits.get(category, other_label_limit))
                    starting_number = int(category_start_numbers.get(category, other_start_number))
                    folder_label_limits[prefix] = label_limit
                    folder_start_numbers[prefix] = starting_number

            # 遍歷每個圖像資料進行角度分配
            for img_data in folder_features:
                image_file = img_data["image_file"]
                special_angles = img_data["special_angles"]
                special_category = img_data["special_category"]
                img_features = img_data["features"]
                best_angle = None

                if special_angles:
                    # 過濾有效的特殊角度（精確匹配）
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
                                index = features_by_category[selected_brand][best_category["category"]]["index"]
                                num_samples = len(features_by_category[selected_brand][best_category["category"]]["labeled_features"])
                                nlist = index.nlist
                                nprobe = max(1, int(np.sqrt(nlist)))
                                index.nprobe = nprobe
                                angle_features = [
                                    item["features"] for item in filtered_by_category 
                                    if item["labels"]["angle"] == angle
                                ]
                                if not angle_features:
                                    continue
                                angle_features = np.array(angle_features, dtype=np.float32)
                                angle_features = l2_normalize(angle_features)
                                temp_index = faiss.IndexFlatIP(angle_features.shape[1])
                                temp_index.add(angle_features)
                                img_query = l2_normalize(img_features.astype(np.float32).reshape(1, -1))
                                similarities, _ = temp_index.search(img_query, k=1)
                                similarity_percentage = similarities[0][0] * 100
                                valid_angles_by_similarity.append(
                                    (angle, similarity_percentage)
                                )
                            
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
                                for idx, rule in enumerate(angle_banning_rules):
                                    if best_angle in rule["if_appears_in_angle"]:
                                        rule_flags[idx] = True
                            else:
                                st.warning(
                                    f"圖片 '{image_file}' 沒有可用的角度可以分配"
                                )
                                final_results[image_file] = None
                                
                                old_image_path = os.path.join(folder_path, image_file)
                                new_image_path = os.path.join("uploaded_images", folder, os.path.basename(image_file))
                                if os.path.exists(old_image_path):
                                    os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                                    os.rename(old_image_path, new_image_path)
                        else:
                            # 只有一個有效的特殊角度
                            special_angle = valid_special_angles[0]
                            if special_angle not in reassigned_allowed and special_angle in used_angles:
                                st.warning(
                                    f"角度 '{special_angle}' 已被使用，圖片 '{image_file}' 無法分配"
                                )
                                final_results[image_file] = None
                                
                                old_image_path = os.path.join(folder_path, image_file)
                                new_image_path = os.path.join("uploaded_images", folder, os.path.basename(image_file))
                                if os.path.exists(old_image_path):
                                    os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                                    os.rename(old_image_path, new_image_path)
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
                                for idx, rule in enumerate(angle_banning_rules):
                                    if special_angle in rule["if_appears_in_angle"]:
                                        rule_flags[idx] = True
                    else:
                        if best_category['category'] == "帽子" and "白背上腳照" in special_angles:
                            old_image_path = os.path.join(folder_path, image_file)
                            new_image_path = os.path.join("uploaded_images", folder, os.path.basename(image_file))
                            if os.path.exists(old_image_path):
                                os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                                os.rename(old_image_path, new_image_path)
                        else:
                            st.warning(
                                f"商品分類 '{best_category['category']}' 中沒有角度 '{', '.join(special_angles)}'，圖片 '{image_file}' 無法分配"
                            )
                            final_results[image_file] = None
                        
                            old_image_path = os.path.join(folder_path, image_file)
                            new_image_path = os.path.join("uploaded_images", folder, os.path.basename(image_file))
                            if os.path.exists(old_image_path):
                                os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                                os.rename(old_image_path, new_image_path)

            non_special_images = [
                img_data for img_data in folder_features 
                if not img_data["special_angles"]
            ]

            if not special_mappings:
                non_special_images = folder_features

            image_similarity_store = {}

            labeled_features = filtered_by_category
            features = np.array([item["features"] for item in labeled_features], dtype=np.float32)
            features = l2_normalize(features)
            labels = [item["labels"] for item in labeled_features]
            index = features_by_category[selected_brand][best_category["category"]]["index"]
            num_samples = len(features_by_category[selected_brand][best_category["category"]]["labeled_features"])
            nlist = index.nlist
            nprobe = max(1, int(np.sqrt(nlist)))
            index.nprobe = nprobe

            for img_data in non_special_images:
                image_file = img_data["image_file"]
                if final_results.get(image_file) is not None:
                    continue

                img_features = img_data["features"].astype(np.float32).reshape(1, -1)
                img_features = l2_normalize(img_features)
                similarities, indices = index.search(img_features, k=len(labels))
                similarities = similarities.flatten()
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

            unassigned_images = set(image_similarity_store.keys())

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
                        
                        if candidate_angle not in reassigned_allowed or candidate_angle not in used_angles:
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
                        used_angles.add(angle)
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
                        used_angles.add(angle)
                        assigned_in_this_round.add(best_image)

                unassigned_images -= assigned_in_this_round
                if not assigned_in_this_round:
                    break

            for image_file, assignment in final_results.items():
                if assignment is not None:
                    results.append(assignment)

            processed_folders += 1
            progress_bar.progress(processed_folders / total_folders)

        progress_bar.empty()
        progress_text.empty()

        results = rename_numbers_in_folder(results, folder_label_limits, folder_start_numbers, angle_to_prefix, category_label_limits, category_start_numbers)

        result_df = pd.DataFrame(results)
        st.dataframe(result_df, hide_index=True, use_container_width=True)
        
        folder_data = []
        for folder in image_folders:
            folder_results = result_df[result_df['資料夾'] == folder]
            valid_images = folder_results[
                (folder_results['編號'] != '超過上限') & (~folder_results['編號'].isna())
            ]
            num_images = len(valid_images)
            if selected_brand == "ADS":
                ad_images = valid_images[valid_images['角度'].str.contains('情境|HM')]
                num_ad_images = len(ad_images)
                if num_ad_images > 0:
                    ad_image_value = f"{num_ad_images + 1:02}"
                else:
                    ad_image_value = "01"
            else:
                ad_image_value = ""
        
            folder_data.append({'資料夾': folder, '張數': num_images, '廣告圖': ad_image_value})
        
        folder_df = pd.DataFrame(folder_data)
        image_type_statistics_df = generate_image_type_statistics(result_df)
        
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, sheet_name='編圖結果', index=False)
            folder_df.to_excel(writer, sheet_name='編圖張數與廣告圖', index=False)
            image_type_statistics_df.to_excel(writer, sheet_name='圖片類型統計', index=False)
        excel_data = excel_buffer.getvalue()

        zip_data = rename_and_zip_folders(results, excel_data, skipped_images, folder_settings, angle_to_prefix)
        if uploaded_zip:
            uploaded_zip_name = os.path.splitext(uploaded_zip.name)[0]  
            download_file_name = f"{uploaded_zip_name}_結果.zip"
        else:
            download_file_name = "結果.zip"
        
        shutil.rmtree("uploaded_images")
        if uploaded_zip:
            os.remove("temp.zip") 
        
        if st.download_button(
            label="下載編圖結果",
            data=zip_data,
            file_name=download_file_name,
            mime="application/zip",
            on_click=reset_key_tab1
        ):
            st.rerun()

#%% 編圖複檢
def initialize_tab2():
    """
    初始化所有 session_state 變數。
    """
    defaults = {
        'filename_changes': {},
        'image_cache': {},
        'folder_values': {},
        'confirmed_changes': {},
        'uploaded_file_name': None,
        'last_text_inputs': {},
        'has_duplicates': False,
        'duplicate_filenames': [],
        'file_uploader_key2': 4,
        'modified_folders': set(),
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

def reset_tab2():
    """
    重置所有 session_state 變數。
    """
    st.session_state['filename_changes'] = {}
    st.session_state['image_cache'] = {}
    st.session_state['folder_values'] = {}
    st.session_state['confirmed_changes'] = {}
    st.session_state['uploaded_file_name'] = None
    st.session_state['last_text_inputs'] = {}
    st.session_state['has_duplicates'] = False
    st.session_state['duplicate_filenames'] = []
    st.session_state['modified_folders'] = set()

def reset_key_tab2():
    """
    重置文件上傳器的狀態，並刪除上傳的圖像和臨時壓縮檔。
    """
    st.session_state['file_uploader_key2'] += 1
    st.session_state['filename_changes'].clear()
    
def get_outer_folder_images(folder_path):
    """
    獲取指定資料夾中所有圖片檔案，並按名稱排序。
    參數:
        folder_path: 資料夾的路徑
    回傳:
        排序後的圖片檔案列表
    """
    return sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    )

def get_prefix(image_files):
    """
    從圖片檔案中取得通用的命名前綴。
    參數:
        image_files: 圖片檔案列表
    回傳:
        圖片檔名的前綴字串（若找不到則回傳空字串）
    """
    for image_file in image_files:
        filename_without_ext = os.path.splitext(image_file)[0]
        first_underscore_index = filename_without_ext.find('_')
        if first_underscore_index != -1:
            return filename_without_ext[:first_underscore_index + 1]
    return ""

def reset_duplicates_flag():
    """
    重設 session state 中的重複檔名標誌。
    """
    st.session_state['has_duplicates'] = False

@functools.lru_cache(maxsize=128)
def load_and_process_image(image_path, add_label=False):
    """
    加載並處理圖片，並加上 "PNG" 標示（如果需要）。
    使用 lru_cache 進行快取以加速重複讀取。
    """
    image = Image.open(image_path)

    # 為 PNG 圖片加上標示
    if add_label and image_path.lower().endswith('.png'):
        image = add_png_label(image)

    # 統一圖片大小為 800x800，保留 ImageOps.pad() 的邏輯
    image = ImageOps.pad(image, (800, 800), method=Image.Resampling.LANCZOS)

    return image

def handle_submission(selected_folder, images_to_display, outer_images_to_display, use_full_filename, folder_to_data):
    """
    處理圖片檔名修改的提交邏輯，包含重命名邏輯與重複檢查。
    參數:
        selected_folder: 當前選擇的資料夾名稱
        images_to_display: 需要顯示的圖片檔案列表（主要處理的圖片）
        outer_images_to_display: 外層資料夾的圖片列表
        use_full_filename: 是否使用完整檔名進行命名
        folder_to_data: 資料夾對應的資料（例如張數和廣告圖）
    """
    current_filenames = {}
    temp_filename_changes = {}
    modified_outer_count = 0  # 記錄修改的 outer images 數量
    removed_image_count = 0  # 記錄 images_to_display 被移除的數量

    # 獲取前綴（僅針對 1-Main/All）
    if not use_full_filename:
        prefix = get_prefix(images_to_display)
        if prefix == "":
            prefix = get_prefix(images_to_display)
    else:
        prefix = ""

    # 處理 images_to_display 的圖片（僅限 1-Main/All）
    for image_file in images_to_display:
        text_input_key = f"{selected_folder}_{image_file}"
        new_text = st.session_state.get(text_input_key, "")

        filename_without_ext = os.path.splitext(image_file)[0]
        extension = os.path.splitext(image_file)[1]

        if not use_full_filename:
            first_underscore_index = filename_without_ext.find('_')
            if first_underscore_index != -1:
                default_text = filename_without_ext[first_underscore_index + 1:]
            else:
                default_text = filename_without_ext
        else:
            default_text = filename_without_ext  # 去掉副檔名

        if new_text.strip() == '':
            new_filename = ''
        else:
            if not use_full_filename:
                new_filename = prefix + new_text + extension
            else:
                new_filename = new_text + extension  # 重新加上副檔名

        current_filenames[image_file] = {'new_filename': new_filename, 'text': new_text}
        temp_filename_changes[image_file] = {'new_filename': new_filename, 'text': new_text}

        # 如果修改後的檔名為空，記錄移除數量
        if new_filename == '':
            removed_image_count += 1

    # 處理 outer_images_to_display 的圖片
    for outer_image_file in outer_images_to_display:
        text_input_key = f"outer_{selected_folder}_{outer_image_file}"
        new_text = st.session_state.get(text_input_key, "")

        filename_without_ext = os.path.splitext(outer_image_file)[0]
        extension = os.path.splitext(outer_image_file)[1]

        if not use_full_filename:
            first_underscore_index = filename_without_ext.find('_')
            if first_underscore_index != -1:
                default_text = filename_without_ext[first_underscore_index + 1:]
            else:
                default_text = filename_without_ext

            if new_text.strip() == '':
                new_filename = ''
            else:
                new_filename = prefix + new_text + extension
        else:
            default_text = filename_without_ext  # 去掉副檔名
            if new_text.strip() == '':
                new_filename = ''
            else:
                new_filename = new_text + extension  # 重新加上副檔名

        if new_text.strip() != default_text:
            current_filenames[outer_image_file] = {'new_filename': new_filename, 'text': new_text}
            temp_filename_changes[outer_image_file] = {'new_filename': new_filename, 'text': new_text}

            # 如果新檔名不為空且有修改，計數修改的 outer images
            if new_filename != '':
                modified_outer_count += 1

    # 檢查重複檔名
    new_filenames = [data['new_filename'] for data in temp_filename_changes.values() if data['new_filename'] != '']
    duplicates = [filename for filename, count in Counter(new_filenames).items() if count > 1]

    if duplicates:
        st.session_state['has_duplicates'] = True
        st.session_state['duplicate_filenames'] = duplicates
        st.session_state['confirmed_changes'][selected_folder] = False
    else:
        st.session_state['has_duplicates'] = False
        st.session_state['confirmed_changes'][selected_folder] = True

        # 僅對 1-Main/All 的圖片進行重新命名
        if not use_full_filename:
            sorted_files = sorted(temp_filename_changes.items(), key=lambda x: x[1]['new_filename'])
            rename_counter = 1

            for file, data in sorted_files:
                if data['new_filename'] != '':  # 忽略空值檔名
                    new_index = str(rename_counter).zfill(2)  # 01, 02, 03 格式
                    extension = os.path.splitext(file)[1]
                    new_filename = f"{prefix}{new_index}{extension}"

                    # 更新 temp_filename_changes 中的檔名
                    temp_filename_changes[file]['new_filename'] = new_filename
                    temp_filename_changes[file]['text'] = new_index

                    rename_counter += 1

        # 更新 session state 的 filename_changes，逐個更新
        if selected_folder not in st.session_state['filename_changes']:
            st.session_state['filename_changes'][selected_folder] = {}
        st.session_state['filename_changes'][selected_folder].update(temp_filename_changes)

        # 更新 text input 顯示
        for file, data in temp_filename_changes.items():
            text_input_key = f"{selected_folder}_{file}"
            st.session_state[text_input_key] = data['text']

    if num_images_key in st.session_state:
        current_num_images = int(st.session_state[num_images_key])
        # 減去被移除的數量，增加修改的 outer images 數量
        st.session_state[num_images_key] = str(max(1, current_num_images - removed_image_count + modified_outer_count))

    ad_images_key = f"{selected_folder}_ad_images"
    ad_images_value = st.session_state.get(ad_images_key)
    data = folder_to_data.get(selected_folder, {})
    data_folder_name = data.get('資料夾', selected_folder)

    # 新增處理模特、平拍、細節的值
    model_images_key = f"{selected_folder}_model_images"
    flat_images_key = f"{selected_folder}_flat_images"
    model_images_value = st.session_state.get(model_images_key)
    flat_images_value = st.session_state.get(flat_images_key)

    st.session_state['folder_values'][data_folder_name] = {
        '張數': st.session_state[num_images_key],
        '廣告圖': ad_images_value,
        '模特': model_images_value,
        '平拍': flat_images_value,
    }

    # 記錄修改過的資料夾
    st.session_state['modified_folders'].add(data_folder_name)

@functools.lru_cache(maxsize=512)
def get_sort_key(image_file):
    """
    根據修改後的檔名取得排序鍵值，用於圖片列表的排序。
    使用 LRU 快取機制加速重複調用，避免多次查詢 session_state。

    參數:
        image_file: 圖片檔案名稱 (str)

    回傳:
        排序鍵值 (str)，若有修改過的檔名則返回修改後的檔名，否則返回原始檔名。
    """
    # 從 session_state 中取得當前資料夾的 filename_changes 字典
    filename_changes = st.session_state.get('filename_changes', {}).get(selected_folder, {})

    # 如果圖片檔名在 filename_changes 中，則返回修改後的檔名
    if image_file in filename_changes:
        new_filename = filename_changes[image_file]['new_filename']
        # 如果修改後的檔名不為空，返回新檔名，否則返回原始檔名
        return new_filename if new_filename else image_file

    # 如果圖片檔名未被修改，則返回原始檔名
    return image_file

def add_png_label(image):
    """
    在圖片右上角加上放大版的 "PNG" 標示，使用實心黑體字。
    參數:
        image: PIL Image 物件
    回傳:
        加上標示後的 Image 物件
    """
    draw = ImageDraw.Draw(image)
    try:
        # 使用系統字體 Arial，大小設為 100
        font = ImageFont.truetype("arial.ttf", 100)  # 請確認系統有安裝 Arial 字體
    except OSError:
        # 如果找不到 Arial 字體，使用 Noto Sans CJK 字體，適合中文系統
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 100)

    text = "PNG"
    # 使用 textbbox 取得文字邊界大小
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    # 設定字樣位置（右上角，留一點內距）
    x = image.width - text_width - 20
    y = 20

    # 使用實心黑色字體
    draw.text((x, y), text, font=font, fill="red")

    return image

with tab2:
    initialize_tab2()
    st.write("\n")
    uploaded_file = st.file_uploader(
        "上傳編圖結果 ZIP 檔",
        type=["zip"],
        key='file_uploader_' + str(st.session_state['file_uploader_key2']),
        on_change=reset_tab2
    )

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
                    sheet_df = pd.DataFrame(columns=['資料夾', '張數', '廣告圖'])
                    folder_to_row_idx = {}

                # 新增讀取 '圖片類型統計' 工作表
                if '圖片類型統計' in excel_sheets:
                    type_sheet_df = excel_sheets['圖片類型統計']
                    type_folder_to_row_idx = {}
                    for idx, row in type_sheet_df.iterrows():
                        folder_name = str(row['資料夾'])
                        type_folder_to_row_idx[folder_name] = idx
                else:
                    type_sheet_df = pd.DataFrame(columns=['資料夾', '模特', '平拍'])
                    type_folder_to_row_idx = {}
            else:
                excel_sheets = {}
                sheet_df = pd.DataFrame(columns=['資料夾', '張數', '廣告圖'])
                folder_to_row_idx = {}
                type_sheet_df = pd.DataFrame(columns=['資料夾', '模特', '平拍'])
                type_folder_to_row_idx = {}

            # 建立資料夾名稱與 '資料夾' 值的對應關係
            folder_to_data = {}
            top_level_folders = [
                name for name in os.listdir(tmpdirname)
                if os.path.isdir(os.path.join(tmpdirname, name)) and not name.startswith(('_', '.'))
            ]

            for folder_name in top_level_folders:
                matched = False
                for data_folder_name in folder_to_row_idx.keys():
                    if data_folder_name in folder_name:
                        idx = folder_to_row_idx[data_folder_name]
                        row = sheet_df.loc[idx]
                        # 新增對 '圖片類型統計' 的處理
                        if data_folder_name in type_folder_to_row_idx:
                            type_idx = type_folder_to_row_idx[data_folder_name]
                            type_row = type_sheet_df.loc[type_idx]
                            folder_to_data[folder_name] = {
                                '資料夾': data_folder_name,
                                '張數': str(row['張數']),
                                '廣告圖': str(row['廣告圖']),
                                '模特': str(type_row['模特']),
                                '平拍': str(type_row['平拍']),
                            }
                        else:
                            folder_to_data[folder_name] = {
                                '資料夾': data_folder_name,
                                '張數': str(row['張數']),
                                '廣告圖': str(row['廣告圖']),
                                '模特': '0',
                                '平拍': '0',
                            }
                        matched = True
                        break
                if not matched:
                    folder_to_data[folder_name] = {
                        '資料夾': folder_name,
                        '張數': '1',
                        '廣告圖': '1',
                        '模特': '0',
                        '平拍': '0',
                    }

            # 初始化 folder_values，確保所有資料夾都有初始值
            for folder_name, data in folder_to_data.items():
                data_folder_name = data.get('資料夾', folder_name)
                if data_folder_name not in st.session_state['folder_values']:
                    st.session_state['folder_values'][data_folder_name] = {
                        '張數': data.get('張數', '1'),
                        '廣告圖': data.get('廣告圖', '1'),
                        '模特': data.get('模特', '0'),
                        '平拍': data.get('平拍', '0'),
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

                # 當 selected_folder 變成 None 時，保存目前的 text_input 值
                if selected_folder is None and previous_folder is not None:
                    st.session_state['last_text_inputs'][previous_folder] = {
                        key: st.session_state[key]
                        for key in st.session_state if key.startswith(f"{previous_folder}_")
                    }

                # 從 None 切回之前的資料夾時，恢復 text_input 值
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

                        # 重建 images_to_display 和 outer_images_to_display
                        all_images = set(image_files + outer_images)

                        images_to_display = []
                        outer_images_to_display = []

                        for image_file in all_images:
                            if selected_folder in st.session_state['filename_changes'] and image_file in st.session_state['filename_changes'][selected_folder]:
                                data = st.session_state['filename_changes'][selected_folder][image_file]
                                if data['new_filename'] == '':
                                    outer_images_to_display.append(image_file)
                                else:
                                    images_to_display.append(image_file)
                            else:
                                if image_file in image_files:
                                    images_to_display.append(image_file)
                                else:
                                    outer_images_to_display.append(image_file)

                        images_to_display.sort(key=get_sort_key)
                        outer_images_to_display.sort(key=get_sort_key)

                        # 建立 basename 與其對應的副檔名列表
                        basename_to_extensions = defaultdict(list)
                        for image_file in all_images:
                            basename, ext = os.path.splitext(image_file)
                            basename_to_extensions[basename].append(ext.lower())

                        with st.form(f"filename_form_{selected_folder}"):
                            cols = st.columns(6)
                            for idx, image_file in enumerate(images_to_display):
                                if idx % 6 == 0 and idx != 0:
                                    cols = st.columns(6)
                                col = cols[idx % 6]

                                # 總是使用原始檔名來讀取圖片
                                image_path = os.path.join(img_folder_path, image_file) if image_file in image_files else os.path.join(outer_folder_path, image_file)
                                if image_path not in st.session_state['image_cache'][selected_folder]:
                                    # 使用快取函數讀取與處理圖片
                                    add_label = image_file.lower().endswith('.png')
                                    image = load_and_process_image(image_path, add_label)

                                    # 將處理後的圖片快取，僅儲存路徑，避免儲存大型圖片物件
                                    st.session_state['image_cache'][selected_folder][image_path] = image
                                else:
                                    image = st.session_state['image_cache'][selected_folder][image_path]

                                col.image(image, use_container_width=True)

                                filename_without_ext = os.path.splitext(image_file)[0]
                                extension = os.path.splitext(image_file)[1]

                                if use_full_filename:
                                    default_text = filename_without_ext  # 去掉副檔名
                                else:
                                    first_underscore_index = filename_without_ext.find('_')
                                    if first_underscore_index != -1:
                                        default_text = filename_without_ext[first_underscore_index + 1:]
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
                            colA,colB,colC,colD,colE = st.columns(5)
                            col1, col2, col3 ,col4= st.columns([1.1,1.71,1.23, 1.23], vertical_alignment="center")
                            if outer_images_to_display:
                                with col4.popover("查看外層圖片"):
                                    outer_cols = st.columns(6)
                                    for idx, outer_image_file in enumerate(outer_images_to_display):
                                        if idx % 6 == 0 and idx != 0:
                                            outer_cols = st.columns(6)
                                        col = outer_cols[idx % 6]

                                        # 總是使用原始檔名來讀取圖片
                                        outer_image_path = os.path.join(outer_folder_path, outer_image_file) if outer_image_file in outer_images else os.path.join(img_folder_path, outer_image_file)

                                        # 使用快取的圖片加載與處理邏輯
                                        if outer_image_path not in st.session_state['image_cache'][selected_folder]:
                                            # 使用快取函數讀取並處理圖片
                                            add_label = outer_image_file.lower().endswith('.png')
                                            outer_image = load_and_process_image(outer_image_path, add_label)

                                            # 儲存處理後的圖片至 session_state 的快取中
                                            st.session_state['image_cache'][selected_folder][outer_image_path] = outer_image
                                        else:
                                            # 直接從快取中取得圖片
                                            outer_image = st.session_state['image_cache'][selected_folder][outer_image_path]

                                        col.image(outer_image, use_container_width=True)

                                        filename_without_ext = os.path.splitext(outer_image_file)[0]
                                        extension = os.path.splitext(outer_image_file)[1]

                                        if use_full_filename:
                                            default_text = filename_without_ext  # 去掉副檔名
                                        else:
                                            first_underscore_index = filename_without_ext.find('_')
                                            if first_underscore_index != -1:
                                                default_text = filename_without_ext[first_underscore_index + 1:]
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

                            if folder_to_data:
                                # 新增張數和廣告圖的選擇框
                                data = folder_to_data.get(selected_folder, {})
                                data_folder_name = data.get('資料夾', selected_folder)
                                if data_folder_name and 'folder_values' in st.session_state and data_folder_name in st.session_state['folder_values']:
                                    num_images_default = st.session_state['folder_values'][data_folder_name]['張數']
                                    ad_images_default = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                    model_images_default = st.session_state['folder_values'][data_folder_name]['模特']
                                    flat_images_default = st.session_state['folder_values'][data_folder_name]['平拍']
                                else:
                                    num_images_default = data.get('張數', '1')
                                    ad_images_default = data.get('廣告圖', '1')
                                    model_images_default = data.get('模特', '0')
                                    flat_images_default = data.get('平拍', '0')
                            
                                num_images_key = f"{selected_folder}_num_images"
                                ad_images_key = f"{selected_folder}_ad_images"
                                model_images_key = f"{selected_folder}_model_images"
                                flat_images_key = f"{selected_folder}_flat_images"
                            
                                if num_images_key not in st.session_state:
                                    st.session_state[num_images_key] = num_images_default
                            
                                if ad_images_key not in st.session_state:
                                    st.session_state[ad_images_key] = ad_images_default
                            
                                if model_images_key not in st.session_state:
                                    st.session_state[model_images_key] = model_images_default
                            
                                if flat_images_key not in st.session_state:
                                    st.session_state[flat_images_key] = flat_images_default
                            
                                # 計算上限
                                upper_limit = max(10, int(num_images_default), int(ad_images_default))
                            
                                num_images_options = [str(i) for i in range(1, upper_limit + 1)]
                                ad_images_options = [str(i) for i in range(1, upper_limit + 1)]
                                type_images_options = [str(i) for i in range(0, 11)]
                            
                                colA.selectbox('張數', num_images_options, key=num_images_key)
                                colB.selectbox('廣告圖', ad_images_options, key=ad_images_key)
                            
                                # 僅在 2-IMG 資料夾存在時顯示
                                if use_full_filename:
                                    colC.selectbox('模特', type_images_options, key=model_images_key)
                                    colD.selectbox('平拍', type_images_options, key=flat_images_key)
                            else:
                                num_images_key = None
                                ad_images_key = None
                                folder_to_data = None


                            col1.form_submit_button(
                                "暫存修改",
                                on_click=handle_submission,
                                args=(selected_folder, images_to_display, outer_images_to_display, use_full_filename, folder_to_data )
                                )
                            if st.session_state.get('has_duplicates') == True:
                                col2.warning(f"檔名重複: {', '.join(st.session_state['duplicate_filenames'])}")

                        if any(st.session_state['confirmed_changes'].values()):
                            if st.checkbox("所有資料夾均完成修改"):
                                with st.spinner('修改檔名中...'):
                                    zip_buffer = BytesIO()
                                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                        # 找出頂層的非資料夾檔案
                                        top_level_files = [name for name in os.listdir(tmpdirname) if os.path.isfile(os.path.join(tmpdirname, name))]

                                        # 先將頂層的非資料夾檔案加入 zip
                                        for file_name in top_level_files:
                                            file_path = os.path.join(tmpdirname, file_name)
                                            arcname = file_name
                                            try:
                                                # 正確寫入文件
                                                if file_name != '編圖結果.xlsx':
                                                    zipf.write(file_path, arcname=arcname)
                                            except Exception as e:
                                                st.error(f"壓縮檔案時發生錯誤：{file_name} - {str(e)}")

                                        # 處理各個資料夾中的檔案
                                        for folder_name in top_level_folders:
                                            folder_path = os.path.join(tmpdirname, folder_name)
                                            for root, dirs, files in os.walk(folder_path):
                                                if "_MACOSX" in root:
                                                    continue
                                                for file in files:
                                                    full_path = os.path.join(root, file)
                                                    rel_path = os.path.relpath(full_path, tmpdirname)
                                                    path_parts = rel_path.split(os.sep)

                                                    original_file = file
                                                    if folder_name in st.session_state['filename_changes'] and original_file in st.session_state['filename_changes'][folder_name]:
                                                        data = st.session_state['filename_changes'][folder_name][original_file]
                                                        new_filename = data['new_filename']
                                                        if new_filename.strip() == '':
                                                            new_rel_path = os.path.join(folder_name, original_file)
                                                        else:
                                                            if use_full_filename:
                                                                idx = path_parts.index(folder_name)
                                                                path_parts = path_parts[:idx+1] + ['2-IMG', new_filename]
                                                            else:
                                                                idx = path_parts.index(folder_name)
                                                                path_parts = path_parts[:idx+1] + ['1-Main', 'All', new_filename]
                                                            new_rel_path = os.path.join(*path_parts)

                                                        try:
                                                            # 檢查是否已經寫入過同樣的路徑，避免重複寫入
                                                            if new_rel_path not in zipf.namelist():
                                                                zipf.write(full_path, arcname=new_rel_path)
                                                        except Exception as e:
                                                            st.error(f"壓縮檔案時發生錯誤：{full_path} - {str(e)}")
                                                    else:
                                                        try:
                                                            zipf.write(full_path, arcname=rel_path)
                                                        except Exception as e:
                                                            st.error(f"壓縮檔案時發生錯誤：{full_path} - {str(e)}")
                                        # 生成 '編圖結果.xlsx' 並加入 zip
                                        excel_buffer = BytesIO()

                                        if excel_sheets:
                                            # 如果上傳的檔案中有 '編圖結果.xlsx'
                                            result_df = excel_sheets.get('編圖張數與廣告圖', pd.DataFrame(columns=['資料夾', '張數', '廣告圖']))

                                            # 更新所有資料夾的 '張數' 和 '廣告圖' 值
                                            for idx, row in result_df.iterrows():
                                                data_folder_name = str(row['資料夾'])
                                                if data_folder_name in st.session_state['folder_values']:
                                                    num_images = st.session_state['folder_values'][data_folder_name]['張數']
                                                    ad_images = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                                    ad_images = f"{int(ad_images):02}"  # 格式化為兩位數字

                                                    result_df.at[idx, '張數'] = num_images
                                                    result_df.at[idx, '廣告圖'] = ad_images

                                            # 如果有新的資料夾沒有在原始 Excel 中，添加它們
                                            existing_folders = set(result_df['資料夾'])
                                            for data_folder_name in st.session_state['folder_values']:
                                                if data_folder_name not in existing_folders:
                                                    num_images = st.session_state['folder_values'][data_folder_name]['張數']
                                                    ad_images = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                                    ad_images = f"{int(ad_images):02}"  # 格式化為兩位數字

                                                    new_row = pd.DataFrame([{
                                                        '資料夾': data_folder_name,
                                                        '張數': num_images,
                                                        '廣告圖': ad_images
                                                    }])
                                                    result_df = pd.concat([result_df, new_row], ignore_index=True)

                                            # 更新 excel_sheets
                                            excel_sheets['編圖張數與廣告圖'] = result_df

                                            # 處理 '圖片類型統計' 工作表
                                            type_result_df = excel_sheets.get('圖片類型統計', pd.DataFrame(columns=['資料夾', '模特', '平拍']))

                                            # 更新所有資料夾的 '模特'、'平拍'、'細節' 值
                                            for idx, row in type_result_df.iterrows():
                                                data_folder_name = str(row['資料夾'])
                                                if data_folder_name in st.session_state['folder_values']:
                                                    model_images = st.session_state['folder_values'][data_folder_name]['模特']
                                                    flat_images = st.session_state['folder_values'][data_folder_name]['平拍']

                                                    type_result_df.at[idx, '模特'] = model_images
                                                    type_result_df.at[idx, '平拍'] = flat_images

                                            # 如果有新的資料夾沒有在原始 Excel 中，添加它們
                                            existing_type_folders = set(type_result_df['資料夾'])
                                            for data_folder_name in st.session_state['folder_values']:
                                                if data_folder_name not in existing_type_folders:
                                                    model_images = st.session_state['folder_values'][data_folder_name]['模特']
                                                    flat_images = st.session_state['folder_values'][data_folder_name]['平拍']

                                                    new_row = pd.DataFrame([{
                                                        '資料夾': data_folder_name,
                                                        '模特': model_images,
                                                        '平拍': flat_images,
                                                    }])
                                                    type_result_df = pd.concat([type_result_df, new_row], ignore_index=True)

                                            # 更新 excel_sheets
                                            excel_sheets['圖片類型統計'] = type_result_df

                                            # 寫入所有工作表
                                            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                                for sheet_name, df in excel_sheets.items():
                                                    df.to_excel(writer, index=False, sheet_name=sheet_name)
                                        else:
                                            # 如果上傳的檔案中沒有 '編圖結果.xlsx'
                                            result_df = pd.DataFrame(columns=['資料夾', '張數', '廣告圖'])
                                            type_result_df = pd.DataFrame(columns=['資料夾', '模特', '平拍'])
                                            for data_folder_name in st.session_state['folder_values']:
                                                num_images = st.session_state['folder_values'][data_folder_name]['張數']
                                                ad_images = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                                ad_images = f"{int(ad_images):02}"  # 格式化為兩位數字
                                                new_row = pd.DataFrame([{
                                                    '資料夾': data_folder_name,
                                                    '張數': num_images,
                                                    '廣告圖': ad_images
                                                }])
                                                result_df = pd.concat([result_df, new_row], ignore_index=True)

                                                model_images = st.session_state['folder_values'][data_folder_name]['模特']
                                                flat_images = st.session_state['folder_values'][data_folder_name]['平拍']
                                                new_type_row = pd.DataFrame([{
                                                    '資料夾': data_folder_name,
                                                    '模特': model_images,
                                                    '平拍': flat_images,
                                                }])
                                                type_result_df = pd.concat([type_result_df, new_type_row], ignore_index=True)
                                            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                                result_df.to_excel(writer, index=False, sheet_name='編圖張數與廣告圖')
                                                type_result_df.to_excel(writer, index=False, sheet_name='圖片類型統計')

                                        # 將 '編圖結果.xlsx' 加入 zip，覆蓋原始檔案
                                        excel_buffer.seek(0)
                                        zipf.writestr('編圖結果.xlsx', excel_buffer.getvalue())

                                    zip_buffer.seek(0)
                                    st.download_button(
                                        label='下載修改後的檔案',
                                        data=zip_buffer,
                                        file_name=uploaded_file.name,
                                        mime='application/zip',
                                        on_click=reset_key_tab2
                                    )

                    else:
                        st.error("未找到圖片。")
                else:
                    st.error("不存在 '2-IMG' 或 '1-Main/All' 資料夾。")
            else:
                st.error("未找到任何資料夾。")

#%% 刪外層圖
with tab3:
    # 支援的圖片檔格式
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
    EXCLUDED_EXTENSIONS = [".xlsx", ".gsheet"]  # 不需要刪除或複製的檔案類型

    if 'file_uploader_key3' not in st.session_state:
        st.session_state['file_uploader_key3'] = 8
    if 'text_area_key3' not in st.session_state:
        st.session_state['text_area_key3'] = 8

    def reset_key_tab3():
        st.session_state['file_uploader_key3'] += 1
        st.session_state['text_area_key3'] += 1

    # 刪除與 "1-Main" 同層的圖片檔案，但保留指定的檔案類型
    def clean_same_level_as_1_Main(root_path):
        for root, dirs, files in os.walk(root_path):
            # 如果資料夾中有 "1-Main"，刪除與其同層的圖片
            if "1-Main" in dirs:
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                        os.remove(file_path)  # 刪除圖片檔案

    # 將處理過的資料夾重新打包為 ZIP 檔案
    def create_zip_from_directory(dir_path):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, dir_path)
                    zip_file.write(file_path, relative_path)
        zip_buffer.seek(0)  # 將指標移到開頭，準備下載
        return zip_buffer

    # 刪除本地目錄內所有檔案，但保留指定的檔案類型
    def clean_local_directory(directory):
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory, topdown=False):
                for file in files:
                    if os.path.splitext(file)[1].lower() in EXCLUDED_EXTENSIONS:
                        continue  # 跳過指定類型的檔案
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))

    # 解壓縮 ZIP 檔案並處理
    def process_zip_and_return(zip_file, local_path=None):
        # 如果 local_path 是 search-ms 格式，嘗試解析並轉換為正確路徑
        if local_path and local_path.startswith("search-ms:"):
            match = re.search(r'location:([^&]+)', local_path)
            if match:
                decoded_path = re.sub(r'%3A', ':', match.group(1))  # 解碼冒號
                decoded_path = re.sub(r'%5C', '\\\\', decoded_path)  # 解碼反斜線
                local_path = decoded_path
            else:
                st.warning("無法解析 search-ms 路徑，將忽略指定的本地路徑")
                local_path = None  # 無法解析時，設定為 None
    
        # 創建唯一的臨時資料夾來解壓縮 ZIP 檔案
        temp_dir = "/tmp/extracted_" + str(st.session_state['file_uploader_key3'])
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)  # 確保臨時資料夾不存在
        os.makedirs(temp_dir, exist_ok=True)
    
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    
        # 如果提供本地路徑，先清空路徑內檔案並複製 ZIP 檔案內容（未處理）到該路徑
        if local_path:
            clean_local_directory(local_path)
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in EXCLUDED_EXTENSIONS:
                        continue  # 跳過指定類型的檔案
                    src = os.path.join(root, file)
                    dst = os.path.join(local_path, os.path.relpath(src, temp_dir))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
    
        # 在臨時目錄中刪除與 "1-Main" 同層的圖片檔案
        clean_same_level_as_1_Main(temp_dir)
    
        # 將處理過的資料夾重新打包成 ZIP
        zip_buffer = create_zip_from_directory(temp_dir)
    
        # 清理臨時資料夾
        shutil.rmtree(temp_dir)
        return zip_buffer


    # 使用者介面
    st.write("\n")
    col1, col2 = st.columns([1.6, 1])
    uploaded_file = col1.file_uploader("上傳複檢完成的 ZIP 檔案", type=["zip"], key='file_uploader_' + str(st.session_state['file_uploader_key3']))
    local_directory = col2.text_area(
        "舊檔案路徑（選填）",
        key='text_area_' + str(st.session_state['text_area_key3']),
        height=78,
        placeholder="會用 ZIP 裡的檔案覆蓋掉所輸入的路徑裡的檔案",
        value=st.session_state.get("input_path_from_tab1", "")  # 使用 Tab1 儲存的路徑作為預設值
    )

    if uploaded_file is not None:
        button_placeholder = st.empty()
        with button_placeholder:
            button_clicked = st.button("開始執行")

        if button_clicked:
            button_placeholder.empty()
            with st.spinner("執行中，請稍候..."):
                # 處理 ZIP 檔案
                processed_zip = process_zip_and_return(BytesIO(uploaded_file.read()), local_path=local_directory.strip() or None)

            # 提供下載選項
            st.write("\n")
            if local_directory:
                st.success("已使用您上傳的檔案覆蓋舊檔案")
            st.download_button(
                label="下載已刪除外層圖片的檔案",
                data=processed_zip,
                file_name=uploaded_file.name.split('.')[0] + "_已刪圖." + uploaded_file.name.split('.')[-1],  # 修改檔案名稱
                mime="application/zip",
                on_click=reset_key_tab3
            )
