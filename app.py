#%% 導入區
import streamlit as st
import pandas as pd
import zipfile
import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageOps, ImageDraw, ImageFont, UnidentifiedImageError
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
import imagecodecs
import ctypes
import subprocess
import sys
from psd_tools import PSDImage

st.set_page_config(page_title='TP自動編圖工具', page_icon='👕', layout="wide")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
faiss.omp_set_num_threads(1)

# 自定義 CSS 以調整頁面樣式
custom_css = """
<style>
section.stMain {
    padding-left: 12%; 
    padding-right: 12%;
}
@media (min-width: 1900px) {
    section.stMain {
        padding-left: 19%;
        padding-right: 19%;
    }
}
@media (max-width: 1400px) {
    section.stMain {
        padding-left: 8%;
        padding-right: 8%;
    }
}
div.stTextInput > label {
    display: none;
}   
div.block-container {
    padding-top: 2rem;
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
div[data-testid=stToast] {
    background-color: #fff8b3;
}
header[data-testid="stHeader"] {
    height: 30px; 
    padding: 5px; 
}
[data-testid="stPopover"] {
    display: flex;
    justify-content: flex-end; 
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

#%% 自動編圖
def find_brand_files(brand_name):
    """
    在指定品牌資料夾下，自動尋找包含 'image_features' 的 pkl 檔與
    包含 '檔名角度對照表' 的 xlsx 檔 (僅會有一個)。
    """
    brand_path = os.path.join("dependencies", brand_name)
    train_file = None
    angle_filename_reference = None
    
    # 在該品牌資料夾尋找指定關鍵字的檔案
    for filename in os.listdir(brand_path):
        lower_filename = filename.lower()
        if lower_filename.endswith(".pkl") and ("image_features" in lower_filename):
            train_file = os.path.join(brand_path, filename)
        if lower_filename.endswith(".xlsx") and ("檔名角度對照表" in lower_filename):
            angle_filename_reference = os.path.join(brand_path, filename)
    
    return train_file, angle_filename_reference

def get_dynamic_nlist(num_samples):
    """
    根據樣本數決定 IVF 的 nlist。
    """
    if num_samples >= 1000:
        return min(200, int(np.sqrt(num_samples)))
    elif num_samples >= 100:
        return min(100, int(np.sqrt(num_samples)))
    else:
        return max(1, num_samples // 2)

def l2_normalize(vectors):
    """
    L2 Normalization。
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def build_ivf_index(features, nlist):
    """
    建立 IVF 索引。
    """
    d = features.shape[1]  
    nlist = min(nlist, len(features))  
    quantizer = faiss.IndexFlatIP(d)  
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(features)
    index.add(features)
    return index

@st.cache_resource
def load_resnet_model():
    """
    載入 ResNet 模型 (去除最後一層分類)。
    """
    device = torch.device("cpu")
    weights_path = "dependencies/resnet50.pt"
    resnet = models.resnet50()
    # 注意：若有 weights_only=True，可能需要依實際需要修改
    resnet.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval().to(device)
    return resnet

@st.cache_resource
def get_preprocess_transforms():
    """
    建立圖像前處理流程。
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
def load_image_features_with_ivf(train_file_path):
    """
    從 pickle 檔載入已標注的特徵資料，並為每個子分類建立 IVF 索引。
    """
    with open(train_file_path, 'rb') as f:
        features_by_category = pickle.load(f)
    
    for brand, categories in features_by_category.items():
        for category, data in categories.items():
            features = np.array([item["features"] for item in data["labeled_features"]], dtype=np.float32)
            features = l2_normalize(features)
            num_samples = len(features)
            nlist = get_dynamic_nlist(num_samples)
            index = build_ivf_index(features, nlist)
            features_by_category[brand][category]["index"] = index
    return features_by_category

def get_image_features(image, model):
    """
    將圖片餵入 ResNet，回傳特徵向量。
    """
    device = torch.device("cpu")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()
    return features

def category_match(image_files, keywords, match_all):
    """
    檢查資料夾是否符合某一分類規則。
    """
    if match_all:
        return all(any(keyword in image_file for image_file in image_files) for keyword in keywords)
    else:
        return any(any(keyword in image_file for image_file in image_files) for keyword in keywords)

def is_banned_angle(item_angle, rule_flags):
    """
    檢查角度是否屬於禁用規則。
    """
    for idx, rule in enumerate(angle_banning_rules):
        if rule_flags[idx]:
            if rule["banned_angle_logic"] == "等於":
                # 若 banned_angle 是列表，檢查是否在其中
                if item_angle in rule["banned_angle"]:
                    return True
            elif rule["banned_angle_logic"] == "包含":
                # 若 banned_angle 是列表，檢查是否包含其中任意值
                if any(banned in item_angle for banned in rule["banned_angle"]):
                    return True
    return False

def generate_image_type_statistics(results):
    """
    產生圖片類型統計（如模特、平拍）。
    """
    filtered_results = results[
        (results["編號"] != "超過上限") & (~results["編號"].isna())
    ]
    statistics = []
    for folder, folder_results in filtered_results.groupby("資料夾"):
        model_count = folder_results["角度"].str.contains("模特").sum()
        excluded_angles = {"HM1", "HM2", "HM3", "HM4", "HM5", "HM6", "HM7", "HM8", "HM9", "HM10"}
        flat_lay_count = folder_results["角度"].apply(
            lambda x: x not in excluded_angles and "模特" not in x 
        ).sum()
        statistics.append({
            "資料夾": folder,
            "模特": model_count,
            "平拍": flat_lay_count,
        })
    return pd.DataFrame(statistics)

def handle_file_uploader_change():
    """
    處理檔案上傳變更邏輯，檢查是否換檔並清空相關暫存。
    """
    file_key = 'file_uploader_' + str(st.session_state.get('file_uploader_key1', 0))
    uploaded_file_1 = st.session_state.get(file_key, None)

    if uploaded_file_1:
        current_filename = uploaded_file_1.name
        if current_filename != st.session_state.get('previous_uploaded_file_name_tab1', None):
            if os.path.exists("uploaded_images"):
                shutil.rmtree("uploaded_images")
            if "custom_tmpdir" in st.session_state and st.session_state["custom_tmpdir"]:
                shutil.rmtree(st.session_state["custom_tmpdir"], ignore_errors=True)
            st.session_state["custom_tmpdir"] = tempfile.mkdtemp()

            st.session_state['previous_uploaded_file_name_tab1'] = current_filename

    st.session_state.text_area_disabled_1 = bool(uploaded_file_1)

def handle_text_area_change():
    """
    處理文字輸入路徑變更邏輯，檢查是否換路徑並清空相關暫存。
    """
    text_key = 'text_area_' + str(st.session_state.get('text_area_key1', 0))
    text_content = st.session_state.get(text_key, "").strip()

    if text_content != st.session_state.get('previous_input_path_tab1', None):
        if os.path.exists("uploaded_images"):
            shutil.rmtree("uploaded_images")
        if "custom_tmpdir" in st.session_state and st.session_state["custom_tmpdir"]:
            shutil.rmtree(st.session_state["custom_tmpdir"], ignore_errors=True)
        st.session_state["custom_tmpdir"] = tempfile.mkdtemp()

        st.session_state['previous_input_path_tab1'] = text_content

    if text_content.startswith("search-ms:"):
        match = re.search(r'location:([^&]+)', text_content)
        if match:
            decoded_path = re.sub(r'%3A', ':', match.group(1))
            decoded_path = re.sub(r'%5C', '\\\\', decoded_path)
            st.session_state[text_key] = decoded_path
        else:
            st.warning("無法解析 search-ms 路徑，請確認輸入格式。")

    st.session_state.file_uploader_disabled_1 = bool(text_content)

def reset_key_tab1():
    """
    重置檔案上傳器與路徑輸入的 key，並解除其 disable 狀態。
    """
    st.session_state['file_uploader_key1'] += 1 
    st.session_state['text_area_key1'] += 1 
    st.session_state['file_uploader_disabled_1'] = False
    st.session_state['text_area_disabled_1'] = False

def unzip_file(uploaded_zip):
    """
    解壓上傳的 zip 檔，並處理編碼問題 (__MACOSX、隱藏檔等)。
    """
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        for member in zip_ref.infolist():
            if "__MACOSX" in member.filename or member.filename.startswith('.'):
                continue
            raw_bytes = member.filename.encode('utf-8', errors='ignore') 
            detected_encoding = chardet.detect(raw_bytes)['encoding']
            try:
                member.filename = raw_bytes.decode(detected_encoding, errors='ignore')
            except (UnicodeDecodeError, LookupError, TypeError):
                member.filename = raw_bytes.decode('utf-8', errors='ignore')
            zip_ref.extract(member, "uploaded_images")

def get_images_in_folder(folder_path):
    """
    取得資料夾內的圖片清單，並判斷是否使用 2-IMG 結構。
    """
    image_files = []
    two_img_folder_path = os.path.join(folder_path, '2-IMG')
    ads_folder_path = os.path.join(folder_path, '1-Main/All')
    use_two_img_folder = False

    if os.path.exists(two_img_folder_path) and os.path.isdir(two_img_folder_path):
        use_two_img_folder = True
        for file in os.listdir(two_img_folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif')):
                full_image_path = os.path.join(two_img_folder_path, file)
                relative_image_path = os.path.relpath(full_image_path, folder_path)
                image_files.append((relative_image_path, full_image_path))
    elif os.path.exists(ads_folder_path) and os.path.isdir(ads_folder_path):
        for file in os.listdir(ads_folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif')):
                full_image_path = os.path.join(ads_folder_path, file)
                relative_image_path = os.path.relpath(full_image_path, folder_path)
                image_files.append((relative_image_path, full_image_path))
    else:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.startswith('.') or os.path.isdir(os.path.join(root, file)):
                    continue
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif')):
                    full_image_path = os.path.join(root, file)
                    relative_image_path = os.path.relpath(full_image_path, folder_path)
                    image_files.append((relative_image_path, full_image_path))

    return image_files, use_two_img_folder

def get_prefix(angle, best_category, folder, angle_to_prefix):
    """
    取得最終 prefix，若該分類設定為 single 且原本 prefix = None，則使用資料夾名稱。
    """
    prefix = angle_to_prefix.get((angle, best_category["category"]), angle_to_prefix.get((angle, None), None))
    cat_setting = category_settings.get(best_category["category"], category_settings.get("其他"))
    if cat_setting["prefix_mode"] == "single" and cat_setting["prefix"] is None:
        prefix = folder
    return prefix
    
def rename_numbers_in_folder(results, category_settings, folder_settings, angle_to_prefix):
    """
    依商品分類設定 (prefix_mode、上限、起始號碼) 將圖檔編號重新命名。
    """
    df = pd.DataFrame(results)
    df["最終前綴"] = df.get("最終前綴", None)

    new_results = []
    for folder in df["資料夾"].unique():
        folder_df = df[df["資料夾"] == folder].copy()
        if len(folder_df) == 0:
            continue
        
        category = folder_df["商品分類"].iloc[0]
        cat_setting = category_settings.get(category, category_settings.get("其他", {
            "prefix_mode": "single",
            "prefix": None,
            "label_limit": 3,
            "start_number": 1
        }))

        folder_df["numeric_編號"] = pd.to_numeric(folder_df["編號"], errors="coerce")
        folder_df.sort_values(by=["numeric_編號"], inplace=True, na_position="last")
        
        if cat_setting["prefix_mode"] == "single":
            prefix = cat_setting["prefix"]
            label_limit = cat_setting["label_limit"]
            start_num = cat_setting["start_number"]
            
            # 如果 "指定前綴" 為 None，則一律用資料夾名稱當前綴
            if prefix is None:
                prefix = folder

            if prefix is None:
                # 仍確保執行邏輯，但其實上面已將 prefix 設為 folder
                valid_idx = folder_df[~folder_df["numeric_編號"].isna()].index
                for i, idx_ in enumerate(valid_idx):
                    if i < label_limit:
                        folder_df.at[idx_, "編號"] = f"{start_num + i:02d}"
                    else:
                        folder_df.at[idx_, "編號"] = "超過上限"
            else:
                valid_idx = folder_df[
                    (~folder_df["numeric_編號"].isna()) & 
                    ((folder_df["最終前綴"] == prefix) | (folder_df["最終前綴"].isna()))
                ].index
                not_match_idx = folder_df[
                    (~folder_df["numeric_編號"].isna()) & 
                    (folder_df["最終前綴"] != prefix) & 
                    (folder_df["最終前綴"].notna())
                ].index
                folder_df.loc[not_match_idx, "編號"] = np.nan

                for i, idx_ in enumerate(valid_idx):
                    if i < label_limit:
                        folder_df.at[idx_, "編號"] = f"{start_num + i:02d}"
                    else:
                        folder_df.at[idx_, "編號"] = "超過上限"
        else:
            prefix_list = cat_setting["prefixes"]
            label_limits = cat_setting["label_limits"]
            start_numbers = cat_setting["start_numbers"]
            
            folder_df.loc[~folder_df["最終前綴"].isin(prefix_list), "編號"] = np.nan
            
            for p_idx, pfx in enumerate(prefix_list):
                pfx_limit = label_limits[p_idx]
                pfx_start = start_numbers[p_idx]
                
                subset_idx = folder_df[
                    (folder_df["最終前綴"] == pfx) & 
                    (~folder_df["numeric_編號"].isna())
                ].index
                for i, idx_ in enumerate(subset_idx):
                    if i < pfx_limit:
                        folder_df.at[idx_, "編號"] = f"{pfx_start + i:02d}"
                    else:
                        folder_df.at[idx_, "編號"] = "超過上限"

        folder_df.drop(columns=["numeric_編號"], inplace=True)
        new_results.append(folder_df)

    new_df = pd.concat(new_results, ignore_index=True)
    return new_df.to_dict('records')

def rename_and_zip_folders(results, output_excel_data, skipped_images, folder_settings, angle_to_prefix):
    """
    根據使用者最終結果 (results)，對圖片重新命名並壓縮下載。
    """
    output_folder_path = "uploaded_images"

    # 原有重新命名邏輯（完整保留）
    for result in results:
        folder_name = result["資料夾"]
        image_file = result["圖片"]
        new_number = result["編號"]
        prefix = result.get("最終前綴", None)

        if prefix is None:
            prefix = folder_name

        folder_path = os.path.join(output_folder_path, folder_name)
        use_two_img_folder = folder_settings.get(folder_name, False)

        if use_two_img_folder:
            main_folder_structure = "2-IMG"
        else:
            main_folder_structure = "1-Main/All"
        main_folder_path = os.path.join(folder_path, main_folder_structure)
        os.makedirs(main_folder_path, exist_ok=True)

        old_image_path = os.path.join(folder_path, image_file)
        file_extension = os.path.splitext(image_file)[1]

        if (use_two_img_folder and (new_number == "超過上限" or pd.isna(new_number))):
            new_image_path = old_image_path
        elif new_number == "超過上限" or pd.isna(new_number):
            new_image_path = os.path.join(folder_path, os.path.basename(image_file))
        else:
            if use_two_img_folder:
                new_image_name = f"{prefix}{new_number}{file_extension}"
            else:
                new_image_name = f"{prefix}_{new_number}{file_extension}"

            new_image_path = os.path.join(main_folder_path, new_image_name)

        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

        if os.path.exists(old_image_path) and old_image_path != new_image_path:
            os.rename(old_image_path, new_image_path)

    # 原有跳過圖片處理邏輯（完整保留）
    for skipped in skipped_images:
        folder_name = skipped["資料夾"]
        image_file = skipped["圖片"]
        folder_path = os.path.join(output_folder_path, folder_name)
        old_image_path = os.path.join(folder_path, image_file)

        use_two_img_folder = folder_settings.get(folder_name, False)
        if not use_two_img_folder:
            new_image_path = os.path.join(folder_path, os.path.basename(image_file))
            if os.path.exists(old_image_path):
                os.rename(old_image_path, new_image_path)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder in os.listdir("uploaded_images"):
            folder_path = os.path.join("uploaded_images", folder)
            if os.path.isdir(folder_path):
                use_two_img_folder = folder_settings.get(folder, False)
                if use_two_img_folder:
                    new_folder_name = folder
                else:
                    new_folder_name = f"{folder}_OK"
                new_folder_path = os.path.join("uploaded_images", new_folder_name)
                os.rename(folder_path, new_folder_path)

                # === 新增保留空資料夾邏輯 ===
                for root, dirs, files in os.walk(new_folder_path):
                    # 手動添加空目錄
                    for dir_name in dirs:
                        dir_full_path = os.path.join(root, dir_name)
                        zip_dir_path = os.path.relpath(dir_full_path, "uploaded_images") + "/"
                        if zip_dir_path not in zipf.namelist():
                            zip_info = zipfile.ZipInfo(zip_dir_path)
                            zip_info.external_attr = (0o755 & 0xFFFF) << 16  # 兼容Windows
                            zipf.writestr(zip_info, b"", zipfile.ZIP_STORED)

                    # 原有檔案寫入邏輯
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, "uploaded_images"))

        zipf.writestr("編圖結果.xlsx", output_excel_data)

    return zip_buffer.getvalue()

tab1, tab2 = st.tabs(["自動編圖", "編圖複檢"])

selected_brand_file = "dependencies/selected_brand.txt"

# 取得所有位於 "dependencies" 下的子資料夾，即為品牌列表
brand_folders = [
    f for f in os.listdir("dependencies")
    if os.path.isdir(os.path.join("dependencies", f)) and not f.startswith('.')
]
brand_list = brand_folders

# 嘗試讀取上次選擇的品牌，如無則預設使用 brand_list[0]（若有需要可自行調整預設值）
if os.path.exists(selected_brand_file):
    with open(selected_brand_file, "r", encoding="utf-8") as f:
        last_selected_brand = f.read().strip()
    if last_selected_brand not in brand_list:
        # 如果檔案中記載的品牌不在目前資料夾中，就以第一個品牌當預設
        last_selected_brand = brand_list[0] if brand_list else ""
else:
    if brand_list:
        last_selected_brand = brand_list[0]
        with open(selected_brand_file, "w", encoding="utf-8") as f:
            f.write(last_selected_brand)
    else:
        # 沒有任何品牌資料夾時就給個空值
        last_selected_brand = ""

with tab1:
    if 'file_uploader_key1' not in st.session_state:
        st.session_state['file_uploader_key1'] = 0
    if 'text_area_key1' not in st.session_state:
        st.session_state['text_area_key1'] = 0
    if 'file_uploader_disabled_1' not in st.session_state:
        st.session_state['file_uploader_disabled_1'] = False
    if 'text_area_disabled_1' not in st.session_state:
        st.session_state['text_area_disabled_1'] = False
    if 'text_area_content' not in st.session_state:
        st.session_state['text_area_content'] = ""
    if 'previous_uploaded_file_name_tab1' not in st.session_state:
        st.session_state['previous_uploaded_file_name_tab1'] = None
    if 'previous_input_path_tab1' not in st.session_state:
        st.session_state['previous_input_path_tab1'] = None
    
    resnet = load_resnet_model()
    preprocess = get_preprocess_transforms()

    st.write("\n")
    col1, col2 = st.columns([1.6, 1])

    uploaded_zip = col1.file_uploader(
        "上傳 ZIP 檔案",
        type=["zip"],
        key='file_uploader_' + str(st.session_state['file_uploader_key1']),
        disabled=st.session_state['file_uploader_disabled_1'],
        on_change=handle_file_uploader_change
    )

    input_path = col2.text_area(
        "或 輸入資料夾路徑",
        height=78,
        key='text_area_' + str(st.session_state['text_area_key1']),
        disabled=st.session_state['text_area_disabled_1'],
        on_change=handle_text_area_change
    )

    start_running = False
    if input_path:
        st.session_state["input_path_from_tab1"] = input_path

    if uploaded_zip or input_path:
        col1_, col2_, col3_ = st.columns([1.5, 2, 2], vertical_alignment="center", gap="medium")
        selectbox_placeholder = col1_.empty()
        button_placeholder = col2_.empty()

        with selectbox_placeholder:
            # 使用先前取得的 brand_list 與 last_selected_brand
            if brand_list:
                # 如果有可用品牌，預設選到上次選擇的品牌
                selected_brand_index = brand_list.index(last_selected_brand) if last_selected_brand in brand_list else 0
                selected_brand = st.selectbox("請選擇品牌", brand_list, index=selected_brand_index, label_visibility="collapsed")
            else:
                # 如果沒有任何品牌資料夾
                selected_brand = st.selectbox("請選擇品牌", [], label_visibility="collapsed")

        if selected_brand != last_selected_brand and selected_brand != "":
            # 使用者切換品牌，更新至檔案
            with open(selected_brand_file, "w", encoding="utf-8") as f:
                f.write(selected_brand)

        with button_placeholder:
            start_running = st.button("開始執行")

        # 只有在使用者點擊「開始執行」時，才去找對應品牌檔案
        if (uploaded_zip or input_path) and start_running:
            if not selected_brand:
                st.error("未偵測到任何品牌資料夾，請確認 'dependencies' 下是否有子資料夾。")
                st.stop()

            # 由我們撰寫的 find_brand_files() 動態找 train_file 與 angle_filename_reference
            train_file, angle_filename_reference = find_brand_files(selected_brand)
            if not train_file or not angle_filename_reference:
                st.error(f"在品牌 {selected_brand} 資料夾中，無法找到 'image_features' pkl 或 '檔名角度對照表' xlsx！")
                st.stop()

            category_settings_df = pd.read_excel(
                angle_filename_reference,
                sheet_name="基本設定",
                usecols=["商品分類", "編圖上限", "編圖起始號碼", "指定前綴"]
            )

            category_settings = {}
            for i, row in category_settings_df.iterrows():
                cat = row["商品分類"]
                limits_str = str(row["編圖上限"])
                starts_str = str(row["編圖起始號碼"])
                prefix_str = str(row["指定前綴"]).strip()

                if ',' in prefix_str:
                    prefix_list = [p.strip() for p in prefix_str.split(',')]
                    limits_list = [int(x.strip()) for x in limits_str.split(',')]
                    starts_list = [int(x.strip()) for x in starts_str.split(',')]
                    category_settings[cat] = {
                        "prefix_mode": "multi",
                        "prefixes": prefix_list,
                        "label_limits": limits_list,
                        "start_numbers": starts_list
                    }
                else:
                    prefix = prefix_str if prefix_str not in ['nan', ''] else None
                    limit = int(limits_str)
                    start_num = int(starts_str)
                    category_settings[cat] = {
                        "prefix_mode": "single",
                        "prefix": prefix,
                        "label_limit": limit,
                        "start_number": start_num
                    }

            if "其他" not in category_settings:
                category_settings["其他"] = {
                    "prefix_mode": "single",
                    "prefix": None,
                    "label_limit": 3,
                    "start_number": 1
                }

            keywords_to_skip = pd.read_excel(
                angle_filename_reference,
                sheet_name='不編的檔名',
                usecols=[0]
            ).iloc[:, 0].dropna().astype(str).tolist()

            substitute_df = pd.read_excel(
                angle_filename_reference,
                sheet_name='有條件使用的檔名',
                usecols=[0, 1]
            )
            substitute = [
                {"set_a": row.iloc[0].split(','), "set_b": row.iloc[1].split(',')}
                for _, row in substitute_df.iterrows()
            ]
            
            reassigned_allowed = pd.read_excel(
                angle_filename_reference,
                sheet_name='可以重複分配的角度',
                usecols=[0]
            ).iloc[:, 0].dropna().tolist()
            
            angle_banning_df = pd.read_excel(
                angle_filename_reference,
                sheet_name='角度禁止規則',
                usecols=[0, 1, 2]
            )
            angle_banning_rules = [
                {
                    "if_appears_in_angle": row.iloc[0].split(','),  # 解析第一欄
                    "banned_angle": row.iloc[1].split(','),         # 解析第二欄為列表
                    "banned_angle_logic": row.iloc[2]
                }
                for _, row in angle_banning_df.iterrows()
            ]

            
            category_rules_df = pd.read_excel(
                angle_filename_reference,
                sheet_name='商品分類及關鍵字條件',
                usecols=[0, 1, 2]
            )
            category_rules = {
                row.iloc[0]: {
                    "keywords": row.iloc[1].split(','),
                    "match_all": row.iloc[2]
                }
                for _, row in category_rules_df.iterrows()
            }
            
            features_by_category = load_image_features_with_ivf(train_file)
            original_features_by_category = {k: v.copy() for k, v in features_by_category.items()}

            selectbox_placeholder.empty()
            button_placeholder.empty()
            
            if os.path.exists("uploaded_images") and os.path.isdir("uploaded_images"):
                shutil.rmtree("uploaded_images")

            if os.path.exists("temp.zip") and os.path.isfile("temp.zip"):
                os.remove("temp.zip")
                
            with st.spinner("讀取檔案中，請稍候..."):
                if uploaded_zip:
                    with open("temp.zip", "wb") as f:
                        f.write(uploaded_zip.getbuffer())
                    unzip_file("temp.zip")
                elif input_path:
                    if not os.path.exists(input_path):
                        st.error("指定的本地路徑不存在，請重新輸入。")
                        st.stop()
                    else:
                        shutil.copytree(input_path, "uploaded_images")

            special_mappings = {}
            angle_to_prefix = {}
            prefix_to_category = {}

            df_angles = pd.read_excel(angle_filename_reference, sheet_name="檔名角度對照表")
            for idx, row in df_angles.iterrows():
                keyword = str(row.iloc[0]).strip()
                category_raw = str(row.iloc[1]).strip()
                if category_raw == 'nan' or category_raw == '':
                    category = None
                    category_filename = None
                else:
                    match = re.match(r'^(.*)\((.*)\)$', category_raw)
                    if match:
                        category = match.group(1).strip()
                        category_filename_raw = match.group(2).strip()
                        category_filename = [x.strip() for x in category_filename_raw.split(',')]
                    else:
                        category = category_raw
                        category_filename = None
                angle = str(row.iloc[2]).strip()
                angles = [a.strip() for a in angle.split(',')]
                prefix = row.iloc[3] if len(row) > 3 and not pd.isna(row.iloc[3]) else None

                special_mappings[keyword] = {
                    'category': category, 
                    'category_filename': category_filename,
                    'angles': angles,
                    'prefix': prefix
                }

                for a in angles:
                    angle_to_prefix[(a, category)] = prefix

                if prefix:
                    prefix_to_category[prefix] = category

            folder_settings = {}
            image_folders = [
                f for f in os.listdir("uploaded_images")
                if os.path.isdir(os.path.join("uploaded_images", f)) 
                and not f.startswith('__MACOSX') 
                and not f.startswith('.')
            ]

            results = []
            skipped_images = []
            progress_bar = st.progress(0)
            progress_text = st.empty()
        
            total_folders = len(image_folders)
            processed_folders = 0
        
            group_conditions = substitute

            for folder in image_folders:
                features_by_category = {k: v.copy() for k, v in original_features_by_category.items()}
        
                folder_path = os.path.join("uploaded_images", folder)
                image_files, use_two_img_folder = get_images_in_folder(folder_path)
                folder_settings[folder] = use_two_img_folder

                if not image_files:
                    st.warning(f"資料夾 {folder} 中沒有有效的圖片，跳過此資料夾")
                    continue
                folder_features = []
        
                progress_text.text(f"正在處理資料夾: {folder}")
        
                special_images = []
                folder_special_category = None

                group_presence = []
                for group in group_conditions:
                    group_presence.append({
                        "set_a_present": False,
                        "set_b_present": False
                    })

                image_filenames = [img[0] for img in image_files]

                for image_file, image_path in image_files:
                    if image_file.startswith('.') or os.path.isdir(image_path):
                        continue
                    for idx, group in enumerate(group_conditions):
                        if any(substr in image_file for substr in group["set_a"]):
                            group_presence[idx]["set_a_present"] = True
                        if any(substr in image_file for substr in group["set_b"]):
                            group_presence[idx]["set_b_present"] = True

                for image_file, image_path in image_files:
                    if image_file.startswith('.') or os.path.isdir(image_path):
                        continue
                    if any(keyword in image_file for keyword in keywords_to_skip):
                        skipped_images.append({"資料夾": folder, "圖片": image_file})
                        continue

                    skip_image = False
                    for idx, group in enumerate(group_conditions):
                        if any(substr in image_file for substr in group["set_b"]):
                            if group_presence[idx]["set_a_present"] and group_presence[idx]["set_b_present"]:
                                skipped_images.append({"資料夾": folder, "圖片": image_file})
                                skip_image = True
                                break
                    if skip_image:
                        continue

                    special_angles = []
                    special_category = None
                    category_filename = None
                    if special_mappings:
                        for substr, mapping in special_mappings.items():
                            if substr in image_file:
                                special_angles = mapping['angles']
                                special_category = mapping['category']
                                category_filename = mapping.get('category_filename')
                                if category_filename:
                                    # 需檢查這些 keyword 是否確實出現在所有檔名
                                    if any(cond in fname for fname in image_filenames for cond in category_filename):
                                        pass
                                    else:
                                        special_category = None
                                if special_category and not folder_special_category:
                                    folder_special_category = special_category
                                break

                    try:
                        img = Image.open(image_path).convert('RGB')
                    except UnidentifiedImageError:
                        with open(image_path, 'rb') as f:
                            raw_data = f.read()
                            decoded_data = imagecodecs.tiff_decode(raw_data)
                            img = Image.fromarray(decoded_data).convert('RGB')

                    img_features = get_image_features(img, resnet)
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

                if len(folder_features) == 0:
                    st.warning(f"資料夾 {folder} 中沒有有效的圖片，跳過此資料夾")
                    continue

                for category, rule in category_rules.items():
                    if category in features_by_category[selected_brand]:
                        if not category_match([file[0] for file in image_files], rule["keywords"], rule["match_all"]):
                            features_by_category[selected_brand].pop(category, None)

                if folder_special_category:
                    best_category = {
                        'brand': selected_brand,
                        'category': folder_special_category
                    }
                else:
                    category_similarities = {}
                    for brand in features_by_category:
                        for category in features_by_category[brand]:
                            index = features_by_category[brand][category]["index"]
                            nlist = index.nlist
                            nprobe = max(1, int(np.sqrt(nlist)))
                            index.nprobe = nprobe
                            folder_similarities = []
                            
                            for img_data in folder_features:
                                img_features = img_data["features"].astype(np.float32).reshape(1, -1)
                                img_features = l2_normalize(img_features)
                                similarities, _ = index.search(img_features, k=5)
                                avg_similarity = np.mean(similarities)
                                folder_similarities.append(avg_similarity)
                            
                            category_similarities[category] = np.mean(folder_similarities)

                    if category_similarities:
                        best_category_name = max(category_similarities, key=category_similarities.get)
                        best_category = {
                            'brand': selected_brand,
                            'category': best_category_name
                        }
                    else:
                        st.warning(f"資料夾 {folder} 無法匹配任何分類，跳過此資料夾")
                        continue

                filtered_by_category = features_by_category[selected_brand][best_category["category"]]["labeled_features"]
                angle_to_number = {
                    item["labels"]["angle"]: item["labels"]["number"] 
                    for item in filtered_by_category
                }
                used_angles = set()
                final_results = {}
                rule_flags = [False for _ in angle_banning_rules]

                # 先處理 special_angles
                for img_data in folder_features:
                    image_file = img_data["image_file"]
                    special_angles = img_data["special_angles"]
                    special_category = img_data["special_category"]
                    if special_angles:
                        angle = special_angles[0]
                        category = special_category if special_category else best_category["category"]
                        specified_prefix = angle_to_prefix.get((angle, category))
                        if specified_prefix is None:
                            specified_prefix = angle_to_prefix.get((angle, None), None)

                # 處理所有 special_images
                for img_data in folder_features:
                    image_file = img_data["image_file"]
                    special_angles = img_data["special_angles"]
                    special_category = img_data["special_category"]
                    img_features = img_data["features"]

                    if special_angles:
                        valid_special_angles = [angle for angle in special_angles if angle in angle_to_number]
                        if valid_special_angles:
                            if len(valid_special_angles) > 1:
                                valid_angles_by_similarity = []
                                for angle in valid_special_angles:
                                    index = features_by_category[selected_brand][best_category["category"]]["index"]
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
                                    valid_angles_by_similarity.append((angle, similarity_percentage))

                                valid_angles_by_similarity.sort(key=lambda x: x[1], reverse=True)
                                chosen_angle = None
                                for angle, similarity_percentage in valid_angles_by_similarity:
                                    if angle not in reassigned_allowed and angle in used_angles:
                                        pass
                                    else:
                                        chosen_angle = angle
                                        best_similarity = similarity_percentage
                                        break
                                
                                if chosen_angle:
                                    prefix = get_prefix(chosen_angle, best_category, folder, angle_to_prefix)
                                    used_angles.add(chosen_angle)
                                    final_results[image_file] = {
                                        "資料夾": folder,
                                        "圖片": image_file,
                                        "商品分類": best_category["category"],
                                        "角度": chosen_angle,
                                        "編號": angle_to_number[chosen_angle],
                                        "最大相似度": f"{best_similarity:.2f}%",
                                        "最終前綴": prefix
                                    }
                                    for idx, rule in enumerate(angle_banning_rules):
                                        if chosen_angle in rule["if_appears_in_angle"]:
                                            rule_flags[idx] = True
                                else:
                                    final_results[image_file] = None
                                    old_image_path = os.path.join(folder_path, image_file)
                                    new_image_path = os.path.join("uploaded_images", folder, os.path.basename(image_file))
                                    if os.path.exists(old_image_path):
                                        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                                        os.rename(old_image_path, new_image_path)
                            else:
                                special_angle = valid_special_angles[0]
                                if special_angle not in reassigned_allowed and special_angle in used_angles:
                                    final_results[image_file] = None
                                    old_image_path = os.path.join(folder_path, image_file)
                                    new_image_path = os.path.join("uploaded_images", folder, os.path.basename(image_file))
                                    if os.path.exists(old_image_path):
                                        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                                        os.rename(old_image_path, new_image_path)
                                else:
                                    prefix = get_prefix(special_angle, best_category, folder, angle_to_prefix)
                                    used_angles.add(special_angle)
                                    final_results[image_file] = {
                                        "資料夾": folder,
                                        "圖片": image_file,
                                        "商品分類": best_category["category"],
                                        "角度": special_angle,
                                        "編號": angle_to_number[special_angle],
                                        "最大相似度": "100.00%",
                                        "最終前綴": prefix
                                    }
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
                                final_results[image_file] = None
                                old_image_path = os.path.join(folder_path, image_file)
                                new_image_path = os.path.join("uploaded_images", folder, os.path.basename(image_file))
                                if os.path.exists(old_image_path):
                                    os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                                    os.rename(old_image_path, new_image_path)

                non_special_images = [img_data for img_data in folder_features if not img_data["special_angles"]]
                if not special_mappings:
                    non_special_images = folder_features

                image_similarity_store = {}
                labeled_features = filtered_by_category
                features = np.array([item["features"] for item in labeled_features], dtype=np.float32)
                features = l2_normalize(features)
                labels = [item["labels"] for item in labeled_features]
                index = features_by_category[selected_brand][best_category["category"]]["index"]
                nlist = index.nlist
                nprobe = max(1, int(np.sqrt(nlist)))
                index.nprobe = nprobe

                # 為 non_special_images 進行相似度匹配
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
                    for idx_, similarity_percentage in zip(indices[0], similarity_percentages):
                        label = labels[idx_]
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
                        if len(unique_labels) == len(labels):
                            break
                    image_similarity_store[image_file] = unique_labels

                unassigned_images = set(image_similarity_store.keys())

                # 依類似度排名將 non_special_images 分配角度
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
                            if candidate_angle not in reassigned_allowed and candidate_angle in used_angles:
                                continue
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
                    for angle, images_ in angle_to_images.items():
                        if angle in reassigned_allowed:
                            for image_file in images_:
                                candidate = image_current_choices[image_file]
                                prefix = get_prefix(angle, candidate["label"], candidate["folder"], angle_to_prefix)
                                final_results[image_file] = {
                                    "資料夾": candidate["folder"],
                                    "圖片": image_file,
                                    "商品分類": candidate["label"]["category"],
                                    "角度": angle,
                                    "編號": candidate["label"]["number"],
                                    "最大相似度": f"{candidate['similarity']:.2f}%",
                                    "最終前綴": prefix
                                }
                                assigned_in_this_round.add(image_file)
                        elif len(images_) == 1:
                            image_file = images_[0]
                            candidate = image_current_choices[image_file]
                            prefix = get_prefix(angle, candidate["label"], candidate["folder"], angle_to_prefix)
                            final_results[image_file] = {
                                "資料夾": candidate["folder"],
                                "圖片": image_file,
                                "商品分類": candidate["label"]["category"],
                                "角度": angle,
                                "編號": candidate["label"]["number"],
                                "最大相似度": f"{candidate['similarity']:.2f}%",
                                "最終前綴": prefix
                            }
                            used_angles.add(angle)
                            assigned_in_this_round.add(image_file)
                        else:
                            max_similarity = -np.inf
                            best_image = None
                            for image_file_ in images_:
                                candidate = image_current_choices[image_file_]
                                if candidate['similarity'] > max_similarity:
                                    max_similarity = candidate['similarity']
                                    best_image = image_file_
                            candidate = image_current_choices[best_image]
                            prefix = get_prefix(angle, candidate["label"], candidate["folder"], angle_to_prefix)
                            final_results[best_image] = {
                                "資料夾": candidate["folder"],
                                "圖片": best_image,
                                "商品分類": candidate["label"]["category"],
                                "角度": angle,
                                "編號": candidate["label"]["number"],
                                "最大相似度": f"{candidate['similarity']:.2f}%",
                                "最終前綴": prefix
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

            results = rename_numbers_in_folder(results, category_settings, folder_settings, angle_to_prefix)

            result_df = pd.DataFrame(results)
            result_df = result_df[result_df['編號'].notna() | (result_df['編號'] == '超過上限')]
            if "最終前綴" in result_df.columns:
                result_df = result_df.drop(columns=["最終前綴"])

            st.dataframe(result_df, hide_index=True, use_container_width=True)
            
            folder_data = []
            for folder in image_folders:
                folder_results = result_df[result_df['資料夾'] == folder]
                valid_images = folder_results[
                    (folder_results['編號'] != '超過上限') & (~folder_results['編號'].isna())
                ]
                num_images = len(valid_images)
                ad_images = valid_images[valid_images['角度'].str.contains('情境')]
                num_ad_images = len(ad_images)
                if num_ad_images > 0:
                    ad_image_value = f"{num_ad_images + 1:02}"
                else:
                    ad_image_value = "01"

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
        'file_uploader_key2': 8,
        'text_area_key2': 6,
        'modified_folders': set(),
        'previous_uploaded_file_name': None,
        'previous_input_path': None,
        'file_uploader_disabled_2': False,
        'text_area_disabled_2': False,
        'input_path_from_tab1': "",
        "custom_tmpdir": tempfile.mkdtemp(),
        'previous_selected_folder': None,
        'final_zip_content': None
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

def setup_temporary_directory(base_path, tmp_dir, read_folder):
    """
    僅複製未被讀取的內容，最後檢查每個最外層資料夾：
    - 如果有 2-IMG，則刪除 2-IMG 資料夾。
    - 如果沒有 2-IMG，但有 1-Main，則刪除 1-Main 資料夾。

    :param base_path: 原始目錄的根路徑
    :param tmp_dir: 臨時目錄路徑
    :param read_folder: 已被讀取的資料夾名稱（如 '2-IMG' 或 '1-Main/All'）
    """
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    for root, dirs, files in os.walk(base_path):
        # 跳過臨時目錄自身，防止無限嵌套
        if tmp_dir in root:
            continue

        # 判斷當前目錄是否與 1-Main 或 2-IMG 同層
        is_top_level = os.path.basename(root) in ['1-Main', '2-IMG']
        is_same_level_as_image_folders = os.path.dirname(root) == base_path

        for item in dirs:
            item_path = os.path.join(root, item)
            relative_path = os.path.relpath(item_path, base_path)

            # 如果是與圖片資料夾同層的資料夾，則複製
            if not (is_same_level_as_image_folders and is_top_level):
                dest_path = os.path.join(tmp_dir, relative_path)
                os.makedirs(dest_path, exist_ok=True)

        for item in files:
            item_path = os.path.join(root, item)
            relative_path = os.path.relpath(item_path, base_path)
            ext = os.path.splitext(item)[1].lower()

            # 如果是與圖片資料夾同層的圖片檔案，則跳過
            if is_same_level_as_image_folders and ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.psd',".ai"]:
                continue

            # 複製其他檔案
            dest_path = os.path.join(tmp_dir, relative_path)
            try:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(item_path, dest_path)
            except FileNotFoundError as e:
                st.warning(f"無法建立路徑：{dest_path}，錯誤：{str(e)}")

    # 最後檢查並刪除不需要的資料夾
    for folder_name in os.listdir(tmp_dir):
        folder_path = os.path.join(tmp_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # 如果包含 2-IMG，刪除 2-IMG
        two_img_path = os.path.join(folder_path, '2-IMG')
        if os.path.exists(two_img_path):
            shutil.rmtree(two_img_path)

        # 如果不包含 2-IMG，但有 1-Main，刪除 1-Main
        elif os.path.exists(os.path.join(folder_path, '1-Main')):
            shutil.rmtree(os.path.join(folder_path, '1-Main'))

def merge_temporary_directory_to_zip(zipf, tmp_dir):
    """
    將臨時目錄中的內容加入 ZIP，保留原始結構。
    :param zipf: ZIP 檔案物件。
    :param tmp_dir: 臨時目錄路徑。
    """
    for root, dirs, files in os.walk(tmp_dir):   
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, tmp_dir)
            zipf.write(file_path, relative_path)

        # 處理空資料夾
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # 空資料夾
                relative_path = os.path.relpath(dir_path, tmp_dir)
                zipf.write(dir_path, relative_path + "/")

def get_outer_folder_images(folder_path):
    """
    獲取指定資料夾中所有圖片檔案，並按名稱排序。
    支援 jpg、jpeg、png、tif、psd (故意排除 .ai)。
    """
    return sorted(
        [
            f for f in os.listdir(folder_path)
            # 加入這行即可確保 .ai 不被視為「要顯示或改名」的檔案
            if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff', 'psd'))
        ]
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

def add_image_label(image, file_extension):
    """
    根據檔案副檔名自動為圖片加上標籤（PNG、TIF 或 PSD）。
    :param image: PIL.Image.Image 物件
    :param file_extension: 檔案副檔名 (如 '.png', '.tif', '.tiff', '.psd')
    :return: 加上標籤後的圖片
    """
    draw = ImageDraw.Draw(image)
    label_map = {'.png': 'PNG', '.tif': 'TIF', '.tiff': 'TIF', '.psd': 'PSD'}

    # 判斷標籤文字
    label_text = label_map.get(file_extension.lower())
    if not label_text:
        return image  # 不支援的格式，直接回傳

    # 設定字體大小
    font_size = max(30, int(image.width * 0.12))
    
    try:
        # 優先嘗試 macOS 系統字體
        if sys.platform == 'darwin':
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
        else:
            font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            # 次選跨平台開源字體
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            # 終極回退使用 Pillow 預設字體
            font = ImageFont.load_default()

    # 文字位置計算
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    # 文字定位 (右上角)
    x = image.width - text_width - 20
    y = 20

    # 添加文字（紅色粗體效果）
    draw.text((x, y), label_text, font=font, fill="red")
    return image

@functools.lru_cache(maxsize=128)
def load_and_process_image(image_path, add_label=False):
    """
    加載並處理圖片，支持 PSD 格式。
    """
    ext = os.path.splitext(image_path)[1].lower()

    # 對 PSD 圖片進行處理
    if ext == '.psd':
        image = PSDImage.open(image_path).composite()
        if image:
            image = image.convert('RGB')  # 確保與其他格式一致
        else:
            raise Exception("無法處理 PSD 文件")
    else:
        try:
            image = Image.open(image_path).convert('RGB')
        except UnidentifiedImageError:
            with open(image_path, 'rb') as f:
                raw_data = f.read()
                decoded_data = imagecodecs.tiff_decode(raw_data)
                image = Image.fromarray(decoded_data).convert('RGB')

    # 添加標籤或調整大小
    if add_label:
        image = add_image_label(image, ext)

    # 統一大小
    image = ImageOps.pad(image, (1000, 1000), method=Image.Resampling.LANCZOS)
    return image

def handle_file_uploader_change_tab2():
    """
    檔案上傳變更時的處理邏輯，檢查是否換檔並清空相關暫存。
    """
    file_key = 'file_uploader_' + str(st.session_state.get('file_uploader_key2', 0))
    uploaded_file_1 = st.session_state.get(file_key, None)

    if uploaded_file_1:
        current_filename = uploaded_file_1.name
        if current_filename != st.session_state['previous_uploaded_file_name']:
            # 清空暫存資料夾
            if "custom_tmpdir" in st.session_state and st.session_state["custom_tmpdir"]:
                shutil.rmtree(st.session_state["custom_tmpdir"], ignore_errors=True)
            st.session_state["custom_tmpdir"] = tempfile.mkdtemp()

            # 清空 image_cache 和其他狀態
            st.session_state['image_cache'].clear()
            st.session_state['filename_changes'].clear()
            st.session_state['confirmed_changes'].clear()
            st.session_state['folder_values'].clear()

            st.session_state['previous_uploaded_file_name'] = current_filename

    # 一旦上傳了檔案，就把 text_area_disabled_2 設為 True
    st.session_state.text_area_disabled_2 = bool(uploaded_file_1)

def handle_text_area_change_tab2():
    """
    處理路徑輸入變更邏輯，檢查是否換路徑並清空相關暫存。
    """
    text_key = 'text_area_' + str(st.session_state.get('text_area_key2', 0))
    text_content = st.session_state.get(text_key, "").strip()

    if text_content != st.session_state['previous_input_path']:
        # 清空暫存資料夾
        if "custom_tmpdir" in st.session_state and st.session_state["custom_tmpdir"]:
            shutil.rmtree(st.session_state["custom_tmpdir"], ignore_errors=True)
        st.session_state["custom_tmpdir"] = tempfile.mkdtemp()

        # 清空 image_cache 和其他狀態
        st.session_state['image_cache'].clear()
        st.session_state['filename_changes'].clear()
        st.session_state['confirmed_changes'].clear()
        st.session_state['folder_values'].clear()

        st.session_state['previous_input_path'] = text_content

    # 一旦輸入了路徑，就把 file_uploader_disabled_2 設為 True
    st.session_state.file_uploader_disabled_2 = bool(text_content)

def get_sort_key(image_file):
    """
    取得排序用 key，若該檔案在 filename_changes 中有新檔名則使用新檔名做排序，否則使用原檔名。
    """
    filename_changes = st.session_state.get('filename_changes', {}).get(selected_folder, {})
    if image_file in filename_changes:
        new_filename = filename_changes[image_file]['new_filename']
        return new_filename if new_filename else image_file
    return image_file

def handle_submission(selected_folder, images_to_display, outer_images_to_display, use_full_filename, folder_to_data):
    """
    處理圖片檔名修改的提交邏輯，包含重命名邏輯與重複檢查。
    """
    current_filenames = {}
    temp_filename_changes = {}
    modified_outer_count = 0
    removed_image_count = 0

    # 取得 prefix
    if not use_full_filename:
        prefix = get_prefix(images_to_display)
    else:
        prefix = ""

    # 依照使用者輸入來更新 new_filename
    for image_file in images_to_display:
        text_input_key = f"{selected_folder}_{image_file}"
        new_text = st.session_state.get(text_input_key, "")

        filename_without_ext = os.path.splitext(image_file)[0]
        extension = os.path.splitext(image_file)[1]

        if not use_full_filename:
            first_underscore_index = filename_without_ext.find('_')
            default_text = (
                filename_without_ext[first_underscore_index + 1:]
                if first_underscore_index != -1 else filename_without_ext
            )
        else:
            default_text = filename_without_ext

        if new_text.strip() == '':
            new_filename = ''
            removed_image_count += 1
        else:
            new_filename = (
                prefix + new_text + extension
                if not use_full_filename else new_text + extension
            )

        current_filenames[image_file] = {'new_filename': new_filename, 'text': new_text}
        temp_filename_changes[image_file] = {'new_filename': new_filename, 'text': new_text}

    # 外層圖片之處理
    for outer_image_file in outer_images_to_display:
        text_input_key = f"outer_{selected_folder}_{outer_image_file}"
        new_text = st.session_state.get(text_input_key, "")

        filename_without_ext = os.path.splitext(outer_image_file)[0]
        extension = os.path.splitext(outer_image_file)[1]

        if not use_full_filename:
            first_underscore_index = filename_without_ext.find('_')
            default_text = (
                filename_without_ext[first_underscore_index + 1:]
                if first_underscore_index != -1 else filename_without_ext
            )

            new_filename = (
                prefix + new_text + extension
                if new_text.strip() != '' else ''
            )
        else:
            default_text = filename_without_ext
            new_filename = (
                new_text + extension if new_text.strip() != '' else ''
            )

        # 只有使用者真的有改動外層檔名時才將其加入 temp_filename_changes
        if new_text.strip() != default_text:
            temp_filename_changes[outer_image_file] = {
                'new_filename': new_filename,
                'text': new_text
            }
            if new_filename != '':
                modified_outer_count += 1

    # 檢查新檔名重複
    new_filenames = [
        data['new_filename']
        for data in temp_filename_changes.values()
        if data['new_filename'] != ''
    ]
    duplicates = [
        filename for filename, count in Counter(new_filenames).items() if count > 1
    ]

    # 若有重複，則設定 has_duplicates
    if duplicates:
        st.session_state['has_duplicates'] = True
        st.session_state['duplicate_filenames'] = duplicates
        st.session_state['confirmed_changes'][selected_folder] = False
    else:
        st.session_state['has_duplicates'] = False
        st.session_state['confirmed_changes'][selected_folder] = True

        # 不使用完整檔名時，需依序重新命名檔名後面的序號
        if not use_full_filename:
            sorted_files = sorted(
                temp_filename_changes.items(),
                key=lambda x: x[1]['new_filename']
            )
            rename_counter = 1
            for file, data in sorted_files:
                if data['new_filename'] != '':
                    new_index = str(rename_counter).zfill(2)
                    extension = os.path.splitext(file)[1]
                    new_filename = f"{prefix}{new_index}{extension}"

                    temp_filename_changes[file]['new_filename'] = new_filename
                    temp_filename_changes[file]['text'] = new_index
                    rename_counter += 1

        # 更新 session_state
        if selected_folder not in st.session_state['filename_changes']:
            st.session_state['filename_changes'][selected_folder] = {}
        st.session_state['filename_changes'][selected_folder].update(temp_filename_changes)

        # 使表單中的 TextInput 與更新後的內容保持同步
        for file, data in temp_filename_changes.items():
            text_input_key = f"{selected_folder}_{file}"
            st.session_state[text_input_key] = data['text']

    # 更新圖片統計數量
    if num_images_key in st.session_state:
        current_num_images = int(st.session_state[num_images_key])
        st.session_state[num_images_key] = str(
            max(1, current_num_images - removed_image_count + modified_outer_count)
        )

    # 取回目前選擇資料夾的統計 key
    ad_images_key = f"{selected_folder}_ad_images"
    ad_images_value = st.session_state.get(ad_images_key)
    model_images_key = f"{selected_folder}_model_images"
    flat_images_key = f"{selected_folder}_flat_images"

    model_images_value = st.session_state.get(model_images_key)
    flat_images_value = st.session_state.get(flat_images_key)
    data = folder_to_data.get(selected_folder, {})
    data_folder_name = data.get('資料夾', selected_folder)

    # 將結果紀錄於 st.session_state['folder_values'] 內
    st.session_state['folder_values'][data_folder_name] = {
        '張數': st.session_state[num_images_key],
        '廣告圖': ad_images_value,
        '模特': model_images_value,
        '平拍': flat_images_value,
    }
    st.session_state['modified_folders'].add(data_folder_name)

def clean_outer_images(zip_buffer):
    """
    從 ZIP buffer 中清理 1-Main 或 2-IMG 同層的圖片，並返回清理後的 ZIP buffer。
    保留所有空資料夾，但排除 tmp_others 資料夾。
    """
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".psd", ".ai"]
    temp_dir = tempfile.mkdtemp()
    cleaned_zip_buffer = BytesIO()

    try:
        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            zip_file.extractall(temp_dir)

        # 清理同層的圖片檔案
        for root, dirs, files in os.walk(temp_dir):
            # 排除 tmp_others 資料夾
            if "tmp_others" in root.split(os.sep):
                continue

            if "1-Main" in dirs or "2-IMG" in dirs:
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                        os.remove(file_path)

        # 重新打包並保留空資料夾 (排除 tmp_others)
        with zipfile.ZipFile(cleaned_zip_buffer, "w", zipfile.ZIP_DEFLATED) as new_zip:
            for root, dirs, files in os.walk(temp_dir):
                # 排除 tmp_others 資料夾及其所有內容
                if "tmp_others" in root.split(os.sep):
                    continue

                # 手動添加空資料夾
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if not os.listdir(dir_path):  # 空資料夾
                        relative_dir = os.path.relpath(dir_path, temp_dir)
                        # 檢查是否為 tmp_others 的子目錄
                        if "tmp_others" not in relative_dir.split(os.sep):
                            zip_info = zipfile.ZipInfo(relative_dir + "/")
                            new_zip.writestr(zip_info, b"")

                # 添加檔案 (排除 tmp_others)
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, temp_dir)
                    if "tmp_others" not in relative_path.split(os.sep):
                        new_zip.write(file_path, arcname=relative_path)
    finally:
        shutil.rmtree(temp_dir)

    cleaned_zip_buffer.seek(0)
    return cleaned_zip_buffer

def cover_path_and_reset_key_tab2():
    """
    重置文件上傳器的狀態，並「使用最終 zip 的檔案」覆蓋指定路徑。
    同時處理可能無法刪除的 .db 檔案，透過終止相關進程並強制刪除。
    確保用來覆蓋的檔案不包含 tmp_others 資料夾。
    """
    if cover_path_input.strip():
        # 清理 tmp_others 資料夾
        tmp_dir_path = st.session_state.get("custom_tmpdir")
        if tmp_dir_path:
            tmp_others_path = os.path.join(tmp_dir_path, "tmp_others")
            if os.path.exists(tmp_others_path):
                shutil.rmtree(tmp_others_path, ignore_errors=True)

        for root, dirs, files in os.walk(cover_path_input, topdown=False):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                file_path = os.path.join(root, file)

                try:
                    # 嘗試刪除檔案
                    if file.lower() == '編圖結果.xlsx':
                        os.remove(file_path)
                    elif ext not in [".xlsx", ".gsheet", ".ai"]:
                        try:
                            os.remove(file_path)
                        except PermissionError:
                            # 若遇到 PermissionError，終止佔用檔案的進程
                            try:
                                if os.name == 'nt':  # Windows 系統
                                    command = f'handle.exe "{file_path}"'
                                    output = subprocess.check_output(command, shell=True, text=True)
                                    for line in output.splitlines():
                                        if "pid:" in line.lower():
                                            pid = int(line.split("pid:")[1].split()[0])
                                            os.system(f"taskkill /PID {pid} /F")
                                else:  # Linux/macOS 系統
                                    command = f'lsof | grep "{file_path}"'
                                    output = subprocess.check_output(command, shell=True, text=True)
                                    for line in output.splitlines():
                                        pid = int(line.split()[1])
                                        os.kill(pid, 9)  # 強制終止進程
                                os.remove(file_path)  # 再次嘗試刪除
                            except Exception as e:
                                st.warning(f"無法刪除檔案: {file_path}，錯誤: {str(e)}")
                except PermissionError:
                    # 使用 ctypes 嘗試解除文件鎖定
                    try:
                        if os.name == 'nt':  # 僅適用於 Windows
                            ctypes.windll.kernel32.SetFileAttributesW(file_path, 0x80)
                            os.remove(file_path)
                    except Exception as e:
                        st.warning(f"無法刪除檔案: {file_path}，錯誤: {str(e)}")

            for d in dirs:
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

        # 從最終 zip 的內容解壓縮到 cover_path_input，確保 tmp_others 不包含在內
        if "final_zip_content" in st.session_state and st.session_state["final_zip_content"]:
            final_zip_bytes = st.session_state["final_zip_content"]
            with zipfile.ZipFile(BytesIO(final_zip_bytes), 'r') as final_zip:
                for zip_info in final_zip.infolist():
                    if not zip_info.filename.startswith("tmp_others/"):
                        ext = os.path.splitext(zip_info.filename)[1].lower()
                        if zip_info.filename.lower().endswith("編圖結果.xlsx"):
                            final_zip.extract(zip_info, cover_path_input)
                        elif ext not in [".xlsx", ".gsheet"]:
                            final_zip.extract(zip_info, cover_path_input)

    if "tmp_dir" in st.session_state and os.path.exists(st.session_state["tmp_dir"]):
        shutil.rmtree(st.session_state["tmp_dir"], ignore_errors=True)

    st.session_state['file_uploader_key2'] += 1
    st.session_state['text_area_key2'] += 1
    st.session_state['file_uploader_disabled_2'] = False
    st.session_state['text_area_disabled_2'] = False
    st.session_state['filename_changes'].clear()

with tab2:
    initialize_tab2()
    st.write("\n")
    col1, col2 = st.columns([1.6, 1])

    uploaded_file_2 = col1.file_uploader(
        "上傳編圖結果 ZIP 檔",
        type=["zip"],
        key='file_uploader_' + str(st.session_state['file_uploader_key2']),
        disabled=st.session_state['file_uploader_disabled_2'],
        on_change=handle_file_uploader_change_tab2
    )
    input_path_2 = col2.text_area(
        "或 輸入資料夾路徑",
        height=78,
        key='text_area_' + str(st.session_state['text_area_key2']),
        disabled=st.session_state['text_area_disabled_2'],
        on_change=handle_text_area_change_tab2
    )

    if uploaded_file_2 or input_path_2:
        tmpdirname = st.session_state["custom_tmpdir"]

        # 檔案來源處理：ZIP 上傳 或 輸入路徑
        if uploaded_file_2:
            with zipfile.ZipFile(uploaded_file_2) as zip_ref:
                zip_ref.extractall(tmpdirname)
        elif input_path_2:
            if input_path_2.startswith("search-ms:"):
                match = re.search(r'location:([^&]+)', input_path_2)
                if match:
                    input_path_2 = re.sub(r'%3A', ':', match.group(1))
                    input_path_2 = re.sub(r'%5C', '\\\\', input_path_2)
                else:
                    st.warning("無法解析 search-ms 路徑，請確認輸入格式。")
            if not os.path.exists(input_path_2):
                st.error("指定的本地路徑不存在，請重新輸入。")
                st.stop()
            else:
                shutil.copytree(input_path_2, tmpdirname, dirs_exist_ok=True)

        # 預先讀取編圖結果.xlsx
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

        # 將頂層資料夾對應到 excel 內的統計資料
        folder_to_data = {}
        top_level_folders = [
            name for name in os.listdir(tmpdirname)
            if os.path.isdir(os.path.join(tmpdirname, name))
            and not name.startswith(('_', '.'))
        ]

        for folder_name in top_level_folders:
            matched = False
            for data_folder_name in folder_to_row_idx.keys():
                if data_folder_name in folder_name:
                    idx = folder_to_row_idx[data_folder_name]
                    row = sheet_df.loc[idx]
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

        # session_state 初始
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
                on_change=lambda: st.session_state.update({'has_duplicates': False})
            )

            # 儲存/取回上一個選擇資料夾的 text_input 狀態
            if selected_folder is None and previous_folder is not None:
                st.session_state['last_text_inputs'][previous_folder] = {
                    key: st.session_state[key]
                    for key in st.session_state if key.startswith(f"{previous_folder}_")
                }

            if selected_folder is not None and previous_folder is None:
                if selected_folder in st.session_state['last_text_inputs']:
                    for key, value in st.session_state['last_text_inputs'][selected_folder].items():
                        st.session_state[key] = value

            st.session_state['previous_selected_folder'] = selected_folder
            st.write("\n")

            if selected_folder is None:
                st.stop()

            # 圖片所在路徑 (優先 2-IMG)
            img_folder_path = os.path.join(tmpdirname, selected_folder, '2-IMG')
            use_full_filename = False

            if not os.path.exists(img_folder_path):
                img_folder_path = os.path.join(tmpdirname, selected_folder, '1-Main', 'All')
                use_full_filename = False
            else:
                use_full_filename = True

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

                    all_images = set(image_files + outer_images)
                    images_to_display = []
                    outer_images_to_display = []

                    # 分類已被使用者「刪名」的圖片 以及 仍在顯示範圍內的圖片
                    for image_file in all_images:
                        if (selected_folder in st.session_state['filename_changes']
                                and image_file in st.session_state['filename_changes'][selected_folder]):
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

                    basename_to_extensions = defaultdict(list)
                    for image_file in all_images:
                        basename, ext = os.path.splitext(image_file)
                        basename_to_extensions[basename].append(ext.lower())

                    with st.form(f"filename_form_{selected_folder}"):
                        colAA, colBB, colCC = st.columns([16.3, 0.01, 2])
                        with colAA:
                            cols = st.columns(6)
                            for idx, image_file in enumerate(images_to_display):
                                if idx % 6 == 0 and idx != 0:
                                    cols = st.columns(6)
                                col = cols[idx % 6]

                                # 圖片顯示
                                image_path = (
                                    os.path.join(img_folder_path, image_file)
                                    if image_file in image_files
                                    else os.path.join(outer_folder_path, image_file)
                                )
                                if image_path not in st.session_state['image_cache'][selected_folder]:
                                    add_label = (
                                        image_file.lower().endswith('.png')
                                        or image_file.lower().endswith('.tif')
                                        or image_file.lower().endswith('.tiff')
                                        or image_file.lower().endswith('.psd')
                                    )
                                    image = load_and_process_image(image_path, add_label)
                                    st.session_state['image_cache'][selected_folder][image_path] = image
                                else:
                                    image = st.session_state['image_cache'][selected_folder][image_path]

                                col.image(image, use_container_width=True)

                                # 預設檔名(不使用完整檔名需排除 prefix)
                                filename_without_ext = os.path.splitext(image_file)[0]
                                extension = os.path.splitext(image_file)[1]

                                if use_full_filename:
                                    default_text = filename_without_ext
                                else:
                                    first_underscore_index = filename_without_ext.find('_')
                                    default_text = (
                                        filename_without_ext[first_underscore_index + 1:]
                                        if first_underscore_index != -1 else filename_without_ext
                                    )

                                if (selected_folder in st.session_state['filename_changes']
                                        and image_file in st.session_state['filename_changes'][selected_folder]):
                                    modified_text = st.session_state['filename_changes'][selected_folder][image_file]['text']
                                else:
                                    modified_text = default_text

                                text_input_key = f"{selected_folder}_{image_file}"
                                if text_input_key not in st.session_state:
                                    st.session_state[text_input_key] = modified_text

                                col.text_input('檔名', key=text_input_key, label_visibility="collapsed")

                            # 選擇張數、廣告圖、模特數、平拍數
                            if folder_to_data:
                                data = folder_to_data.get(selected_folder, {})
                                data_folder_name = data.get('資料夾', selected_folder)
                                if (data_folder_name
                                        and 'folder_values' in st.session_state
                                        and data_folder_name in st.session_state['folder_values']):
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

                                upper_limit = max(
                                    10,
                                    int(num_images_default),
                                    int(ad_images_default)
                                )

                                num_images_options = [str(i) for i in range(1, upper_limit + 1)]
                                ad_images_options = [str(i) for i in range(1, upper_limit + 1)]
                                type_images_options = [str(i) for i in range(0, 11)]

                                with colCC:
                                    st.selectbox('張數', num_images_options, key=num_images_key)
                                    st.selectbox('廣告圖', ad_images_options, key=ad_images_key)
                                    st.selectbox('模特數', type_images_options, key=model_images_key)
                                    st.selectbox('平拍數', type_images_options, key=flat_images_key)
                            else:
                                num_images_key = None
                                ad_images_key = None
                                folder_to_data = None

                        st.divider()
                        colA, colB, colC, colD = st.columns([3, 7, 2, 2.5], vertical_alignment="center")

                        # 暫存修改
                        if colA.form_submit_button(
                            "暫存修改",
                            on_click=handle_submission,
                            args=(selected_folder, images_to_display, outer_images_to_display, use_full_filename, folder_to_data)
                        ):
                            if st.session_state.get('has_duplicates') is False:
                                st.toast(f"資料夾 {selected_folder} 暫存修改成功!", icon='🎉')

                        # 外層圖片 popover
                        if outer_images_to_display:
                            with colD.popover("外層圖片"):
                                outer_cols = st.columns(6)
                                for idx, outer_image_file in enumerate(outer_images_to_display):
                                    if idx % 6 == 0 and idx != 0:
                                        outer_cols = st.columns(6)
                                    col = outer_cols[idx % 6]

                                    outer_image_path = (
                                        os.path.join(outer_folder_path, outer_image_file)
                                        if outer_image_file in outer_images
                                        else os.path.join(img_folder_path, outer_image_file)
                                    )

                                    if (outer_image_path
                                            not in st.session_state['image_cache'][selected_folder]):
                                        add_label = (
                                            outer_image_file.lower().endswith('.png')
                                            or outer_image_file.lower().endswith('.tif')
                                            or outer_image_file.lower().endswith('.tiff')
                                        )
                                        outer_image = load_and_process_image(outer_image_path, add_label)
                                        st.session_state['image_cache'][selected_folder][outer_image_path] = outer_image
                                    else:
                                        outer_image = st.session_state['image_cache'][selected_folder][outer_image_path]

                                    col.image(outer_image, use_container_width=True)

                                    filename_without_ext = os.path.splitext(outer_image_file)[0]
                                    extension = os.path.splitext(outer_image_file)[1]

                                    if use_full_filename:
                                        default_text = filename_without_ext
                                    else:
                                        first_underscore_index = filename_without_ext.find('_')
                                        default_text = (
                                            filename_without_ext[first_underscore_index + 1:]
                                            if first_underscore_index != -1 else filename_without_ext
                                        )

                                    if (selected_folder in st.session_state['filename_changes']
                                            and outer_image_file in st.session_state['filename_changes'][selected_folder]):
                                        modified_text = st.session_state['filename_changes'][selected_folder][outer_image_file]['text']
                                        if modified_text == '':
                                            modified_text = st.session_state['filename_changes'][selected_folder][outer_image_file].get(
                                                'last_non_empty',
                                                default_text
                                            )
                                    else:
                                        modified_text = default_text

                                    text_input_key = f"outer_{selected_folder}_{outer_image_file}"
                                    col.text_input('檔名', value=modified_text, key=text_input_key)

                        # 若有檔名重複
                        if st.session_state.get('has_duplicates'):
                            colB.warning(f"檔名重複: {', '.join(st.session_state['duplicate_filenames'])}")

                    # 所有資料夾均確認完成
                    if st.checkbox("所有資料夾均確認完成"):
                        with st.spinner('修改檔名中...'):
                            # 設定臨時目錄路徑
                            tmp_dir_for_others = os.path.join(tmpdirname, "tmp_others")
                            st.session_state["tmp_dir"] = tmp_dir_for_others  # 記錄到 session_state

                            # 掃描非圖片檔案與資料夾，複製到臨時目錄
                            image_folder = "2-IMG" if os.path.exists(os.path.join(tmpdirname, "2-IMG")) else os.path.join("1-Main", "All")
                            setup_temporary_directory(tmpdirname, tmp_dir_for_others, image_folder)

                            # ** 新增偵測 PSD 檔 **
                            contains_psd = False
                            for folder_name in top_level_folders:
                                folder_path = os.path.join(tmpdirname, folder_name)
                                for root, dirs, files in os.walk(folder_path):
                                    if any(f.lower().endswith('.psd') for f in files):
                                        contains_psd = True
                                        break
                                if contains_psd:
                                    break

                            zip_buffer = BytesIO()

                            # 情況 A：無 psd => 壓縮 + clean_outer_images + download_button
                            if not contains_psd:
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                    # 先壓縮頂層檔案
                                    top_level_files = [
                                        name for name in os.listdir(tmpdirname)
                                        if os.path.isfile(os.path.join(tmpdirname, name))
                                    ]
                                    for file_name in top_level_files:
                                        file_path = os.path.join(tmpdirname, file_name)
                                        arcname = file_name
                                        try:
                                            if file_name != '編圖結果.xlsx':
                                                zipf.write(file_path, arcname=arcname)
                                        except Exception as e:
                                            st.error(f"壓縮檔案時發生錯誤：{file_name} - {str(e)}")

                                    # 壓縮所有資料夾，並將經過修改後的檔名對應寫入
                                    for folder_name in top_level_folders:
                                        folder_path = os.path.join(tmpdirname, folder_name)
                                        for root, dirs, files in os.walk(folder_path):
                                            if "_MACOSX" in root or tmp_dir_for_others in root:
                                                continue
                                            for file in files:
                                                full_path = os.path.join(root, file)
                                                rel_path = os.path.relpath(full_path, tmpdirname)
                                                path_parts = rel_path.split(os.sep)

                                                original_file = file
                                                if (folder_name in st.session_state['filename_changes']
                                                        and original_file in st.session_state['filename_changes'][folder_name]):
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
                                                        if new_rel_path not in zipf.namelist():
                                                            zipf.write(full_path, arcname=new_rel_path)
                                                    except Exception as e:
                                                        st.error(f"壓縮檔案時發生錯誤：{full_path} - {str(e)}")
                                                else:
                                                    try:
                                                        zipf.write(full_path, arcname=rel_path)
                                                    except Exception as e:
                                                        st.error(f"壓縮檔案時發生錯誤：{full_path} - {str(e)}")

                                    # 合併臨時目錄到 ZIP
                                    merge_temporary_directory_to_zip(zipf, tmp_dir_for_others)

                                    # 更新/寫回 編圖結果.xlsx
                                    excel_buffer = BytesIO()
                                    if excel_sheets:
                                        result_df = excel_sheets.get(
                                            '編圖張數與廣告圖',
                                            pd.DataFrame(columns=['資料夾', '張數', '廣告圖'])
                                        )
                                        for idx, row in result_df.iterrows():
                                            data_folder_name = str(row['資料夾'])
                                            if data_folder_name in st.session_state['folder_values']:
                                                num_images = st.session_state['folder_values'][data_folder_name]['張數']
                                                ad_images = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                                ad_images = f"{int(ad_images):02}"

                                                result_df.at[idx, '張數'] = num_images
                                                result_df.at[idx, '廣告圖'] = ad_images

                                        existing_folders = set(result_df['資料夾'])
                                        for data_folder_name in st.session_state['folder_values']:
                                            if data_folder_name not in existing_folders:
                                                num_images = st.session_state['folder_values'][data_folder_name]['張數']
                                                ad_images = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                                ad_images = f"{int(ad_images):02}"

                                                new_row = pd.DataFrame([{
                                                    '資料夾': data_folder_name,
                                                    '張數': num_images,
                                                    '廣告圖': ad_images
                                                }])
                                                result_df = pd.concat([result_df, new_row], ignore_index=True)

                                        excel_sheets['編圖張數與廣告圖'] = result_df

                                        type_result_df = excel_sheets.get(
                                            '圖片類型統計',
                                            pd.DataFrame(columns=['資料夾', '模特', '平拍'])
                                        )
                                        for idx, row in type_result_df.iterrows():
                                            data_folder_name = str(row['資料夾'])
                                            if data_folder_name in st.session_state['folder_values']:
                                                model_images = st.session_state['folder_values'][data_folder_name]['模特']
                                                flat_images = st.session_state['folder_values'][data_folder_name]['平拍']

                                                type_result_df.at[idx, '模特'] = model_images
                                                type_result_df.at[idx, '平拍'] = flat_images

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

                                        excel_sheets['圖片類型統計'] = type_result_df

                                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                            for sheet_name, df in excel_sheets.items():
                                                df.to_excel(writer, index=False, sheet_name=sheet_name)
                                    else:
                                        result_df = pd.DataFrame(columns=['資料夾', '張數', '廣告圖'])
                                        type_result_df = pd.DataFrame(columns=['資料夾', '模特', '平拍'])

                                        for data_folder_name in st.session_state['folder_values']:
                                            num_images = st.session_state['folder_values'][data_folder_name]['張數']
                                            ad_images = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                            ad_images = f"{int(ad_images):02}"
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

                                    excel_buffer.seek(0)
                                    zipf.writestr('編圖結果.xlsx', excel_buffer.getvalue())

                                zip_buffer.seek(0)
                                st.session_state["final_zip_content"] = zip_buffer.getvalue()

                                # 執行 clean_outer_images
                                cleaned_zip_buffer = clean_outer_images(zip_buffer)

                                # 命名下載檔案
                                if uploaded_file_2:
                                    download_file_name = uploaded_file_2.name.replace(".zip", "_已複檢.zip")
                                elif input_path_2:
                                    folder_name = os.path.basename(input_path_2.strip(os.sep))
                                    download_file_name = f"{folder_name}__已複檢.zip"
                                else:
                                    download_file_name = "結果_已複檢.zip"

                                # 仍使用 download_button
                                col1_, col2_ = st.columns([2.7, 1],vertical_alignment="center")
                                if st.session_state["input_path_from_tab1"]:
                                    cover_text_default = st.session_state.get("input_path_from_tab1")
                                elif not uploaded_file_2 and input_path_2:
                                    cover_text_default = input_path_2.strip()
                                else:
                                    cover_text_default = ""

                                cover_path_input = col1_.text_input(
                                    label="同步覆蓋此路徑的檔案",
                                    value=cover_text_default,
                                    placeholder="   輸入分包資料夾路徑以直接覆蓋原檔案 (選填)",
                                )
                                col2_.download_button(
                                    label='下載修改後的檔案',
                                    data=cleaned_zip_buffer,
                                    file_name=download_file_name,
                                    mime='application/zip',
                                    on_click=cover_path_and_reset_key_tab2
                                )

                            # 情況 B：有 psd => 改用 ZIP_STORED、不呼叫 clean_outer_images，改用 st.button
                            else:
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_STORED) as zipf:
                                    # 先壓縮頂層檔案
                                    top_level_files = [
                                        name for name in os.listdir(tmpdirname)
                                        if os.path.isfile(os.path.join(tmpdirname, name))
                                    ]
                                    for file_name in top_level_files:
                                        file_path = os.path.join(tmpdirname, file_name)
                                        arcname = file_name
                                        try:
                                            if file_name != '編圖結果.xlsx':
                                                zipf.write(file_path, arcname=arcname)
                                        except Exception as e:
                                            st.error(f"壓縮檔案時發生錯誤：{file_name} - {str(e)}")

                                    # 壓縮所有資料夾，並將經過修改後的檔名對應寫入
                                    for folder_name in top_level_folders:
                                        folder_path = os.path.join(tmpdirname, folder_name)
                                        for root, dirs, files in os.walk(folder_path):
                                            if "_MACOSX" in root or tmp_dir_for_others in root:
                                                continue
                                            for file in files:
                                                full_path = os.path.join(root, file)
                                                rel_path = os.path.relpath(full_path, tmpdirname)
                                                path_parts = rel_path.split(os.sep)

                                                original_file = file
                                                if (folder_name in st.session_state['filename_changes']
                                                        and original_file in st.session_state['filename_changes'][folder_name]):
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
                                                        if new_rel_path not in zipf.namelist():
                                                            zipf.write(full_path, arcname=new_rel_path)
                                                    except Exception as e:
                                                        st.error(f"壓縮檔案時發生錯誤：{full_path} - {str(e)}")
                                                else:
                                                    try:
                                                        zipf.write(full_path, arcname=rel_path)
                                                    except Exception as e:
                                                        st.error(f"壓縮檔案時發生錯誤：{full_path} - {str(e)}")

                                    # 合併臨時目錄到 ZIP
                                    merge_temporary_directory_to_zip(zipf, tmp_dir_for_others)

                                    # 更新/寫回 編圖結果.xlsx
                                    excel_buffer = BytesIO()
                                    if excel_sheets:
                                        result_df = excel_sheets.get(
                                            '編圖張數與廣告圖',
                                            pd.DataFrame(columns=['資料夾', '張數', '廣告圖'])
                                        )
                                        for idx, row in result_df.iterrows():
                                            data_folder_name = str(row['資料夾'])
                                            if data_folder_name in st.session_state['folder_values']:
                                                num_images = st.session_state['folder_values'][data_folder_name]['張數']
                                                ad_images = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                                ad_images = f"{int(ad_images):02}"

                                                result_df.at[idx, '張數'] = num_images
                                                result_df.at[idx, '廣告圖'] = ad_images

                                        existing_folders = set(result_df['資料夾'])
                                        for data_folder_name in st.session_state['folder_values']:
                                            if data_folder_name not in existing_folders:
                                                num_images = st.session_state['folder_values'][data_folder_name]['張數']
                                                ad_images = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                                ad_images = f"{int(ad_images):02}"

                                                new_row = pd.DataFrame([{
                                                    '資料夾': data_folder_name,
                                                    '張數': num_images,
                                                    '廣告圖': ad_images
                                                }])
                                                result_df = pd.concat([result_df, new_row], ignore_index=True)

                                        excel_sheets['編圖張數與廣告圖'] = result_df

                                        type_result_df = excel_sheets.get(
                                            '圖片類型統計',
                                            pd.DataFrame(columns=['資料夾', '模特', '平拍'])
                                        )
                                        for idx, row in type_result_df.iterrows():
                                            data_folder_name = str(row['資料夾'])
                                            if data_folder_name in st.session_state['folder_values']:
                                                model_images = st.session_state['folder_values'][data_folder_name]['模特']
                                                flat_images = st.session_state['folder_values'][data_folder_name]['平拍']

                                                type_result_df.at[idx, '模特'] = model_images
                                                type_result_df.at[idx, '平拍'] = flat_images

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

                                        excel_sheets['圖片類型統計'] = type_result_df

                                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                            for sheet_name, df in excel_sheets.items():
                                                df.to_excel(writer, index=False, sheet_name=sheet_name)
                                    else:
                                        result_df = pd.DataFrame(columns=['資料夾', '張數', '廣告圖'])
                                        type_result_df = pd.DataFrame(columns=['資料夾', '模特', '平拍'])

                                        for data_folder_name in st.session_state['folder_values']:
                                            num_images = st.session_state['folder_values'][data_folder_name]['張數']
                                            ad_images = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                            ad_images = f"{int(ad_images):02}"
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

                                    excel_buffer.seek(0)
                                    zipf.writestr('編圖結果.xlsx', excel_buffer.getvalue())

                                zip_buffer.seek(0)
                                st.session_state["final_zip_content"] = zip_buffer.getvalue()

                                # 命名下載檔案(維持原邏輯)
                                if uploaded_file_2:
                                    download_file_name = uploaded_file_2.name.replace(".zip", "_已複檢.zip")
                                elif input_path_2:
                                    folder_name = os.path.basename(input_path_2.strip(os.sep))
                                    download_file_name = f"{folder_name}__已複檢.zip"
                                else:
                                    download_file_name = "結果_已複檢.zip"

                                # 改用 st.button，點擊時覆蓋
                                col1_, col2_ = st.columns([2.7, 1],vertical_alignment="center")
                                if st.session_state["input_path_from_tab1"]:
                                    cover_text_default = st.session_state.get("input_path_from_tab1")
                                elif not uploaded_file_2 and input_path_2:
                                    cover_text_default = input_path_2.strip()
                                else:
                                    cover_text_default = ""

                                cover_path_input = col1_.text_input(
                                    label="同步覆蓋此路徑的檔案",
                                    value=cover_text_default,
                                    placeholder="   輸入分包資料夾路徑以直接覆蓋原檔案 (選填)",
                                )
                                col2_.button(
                                    label='覆蓋舊檔案',
                                    on_click=cover_path_and_reset_key_tab2
                                )
                else:
                    st.error("未找到圖片。")
            else:
                st.error("不存在 '2-IMG' 或 '1-Main/All' 資料夾。")
        else:
            st.error("未找到任何資料夾。")
