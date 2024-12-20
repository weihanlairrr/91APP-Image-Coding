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

st.markdown(custom_css, unsafe_allow_html=True)

#%% function
@st.cache_resource
def load_resnet_model():
    device = torch.device("cpu")
    weights_path = "dependencies/resnet50.pt"
    resnet = models.resnet50()
    resnet.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval().to(device)
    return resnet

@st.cache_resource
def get_preprocess_transforms():
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
    if num_samples >= 1000:
        return min(200, int(np.sqrt(num_samples)))  
    elif num_samples >= 100:
        return min(100, int(np.sqrt(num_samples)))  
    else:
        return max(1, num_samples // 2)  

@st.cache_resource
def load_image_features_with_ivf(train_file_path):
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

def build_ivf_index(features, nlist):
    d = features.shape[1]  
    nlist = min(nlist, len(features))  
    quantizer = faiss.IndexFlatIP(d)  
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(features)  
    index.add(features)  
    return index

def get_image_features(image, model):
    device = torch.device("cpu")
    image = preprocess(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()  
    return features

def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def handle_file_uploader_change():
    file_key = 'file_uploader_' + str(st.session_state.get('file_uploader_key1', 0))
    uploaded_file = st.session_state.get(file_key, None)
    st.session_state.text_area_disabled = bool(uploaded_file)

def handle_text_area_change():
    text_key = 'text_area_' + str(st.session_state.get('text_area_key1', 0))
    text_content = st.session_state.get(text_key, "")
    if text_content.startswith("search-ms:"):
        match = re.search(r'location:([^&]+)', text_content)
        if match:
            decoded_path = re.sub(r'%3A', ':', match.group(1)) 
            decoded_path = re.sub(r'%5C', '\\\\', decoded_path) 
            st.session_state[text_key] = decoded_path
        else:
            st.warning("無法解析 search-ms 路徑，請確認輸入格式。")
    st.session_state.file_uploader_disabled = bool(st.session_state[text_key])
            
def reset_key_tab1():
    st.session_state['file_uploader_key1'] += 1 
    st.session_state['text_area_key1'] += 1 
    st.session_state['file_uploader_disabled'] = False
    st.session_state['text_area_disabled'] = False

def unzip_file(uploaded_zip):
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

def rename_numbers_in_folder(results, category_settings, folder_settings, angle_to_prefix):
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
            
            # 如果"指定前綴"為None(空值)，則忽略angle_filename_reference的prefix，一律用資料夾名稱
            if prefix is None:
                prefix = folder

            if prefix is None:
                valid_idx = folder_df[~folder_df["numeric_編號"].isna()].index
                for i, idx_ in enumerate(valid_idx):
                    if i < label_limit:
                        folder_df.at[idx_, "編號"] = f"{start_num + i:02d}"
                    else:
                        folder_df.at[idx_, "編號"] = "超過上限"
            else:
                valid_idx = folder_df[(~folder_df["numeric_編號"].isna()) & ((folder_df["最終前綴"] == prefix) | (folder_df["最終前綴"].isna()))].index
                not_match_idx = folder_df[(~folder_df["numeric_編號"].isna()) & (folder_df["最終前綴"] != prefix) & (folder_df["最終前綴"].notna())].index
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
                
                subset_idx = folder_df[(folder_df["最終前綴"] == pfx) & (~folder_df["numeric_編號"].isna())].index
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
    output_folder_path = "uploaded_images"

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

        if (
            use_two_img_folder
            and (new_number == "超過上限" or pd.isna(new_number))
        ):
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
                for root, dirs, files in os.walk(new_folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, "uploaded_images"))
        zipf.writestr("編圖結果.xlsx", output_excel_data)

    return zip_buffer.getvalue()

def category_match(image_files, keywords, match_all):
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
    filtered_results = results[(results["編號"] != "超過上限") & (~results["編號"].isna())]
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

    uploaded_zip = col1.file_uploader(
        "上傳 ZIP 檔案",
        type=["zip"],
        key='file_uploader_' + str(st.session_state['file_uploader_key1']),
        disabled=st.session_state['file_uploader_disabled'],
        on_change=handle_file_uploader_change
    )

    input_path = col2.text_area(
        "或 輸入資料夾路徑",
        height=78,
        key='text_area_' + str(st.session_state['text_area_key1']),
        disabled=st.session_state['text_area_disabled'],
        on_change=handle_text_area_change
    )

    start_running = False
    if input_path:
        st.session_state["input_path_from_tab1"] = input_path

    if uploaded_zip or input_path:
        col1, col2, col3 = st.columns([1.5, 2, 2], vertical_alignment="center", gap="medium")
        selectbox_placeholder = col1.empty()
        button_placeholder = col2.empty()
        with selectbox_placeholder:
            selected_brand = st.selectbox(
                "請選擇品牌", brand_list, label_visibility="collapsed")
        with button_placeholder:
            start_running = st.button("開始執行")

        dependencies = brand_dependencies[selected_brand]
        train_file = dependencies["train_file"]
        angle_filename_reference = dependencies["angle_filename_reference"]
        
        category_settings_df = pd.read_excel(angle_filename_reference, sheet_name="基本設定", usecols=["商品分類","編圖上限","編圖起始號碼","指定前綴"])

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

        keywords_to_skip = pd.read_excel(angle_filename_reference, sheet_name='不編的檔名', usecols=[0]).iloc[:, 0].dropna().astype(str).tolist()
        substitute_df = pd.read_excel(angle_filename_reference, sheet_name='有條件使用的檔名', usecols=[0, 1])
        substitute = [{"set_a": row.iloc[0].split(','), "set_b": row.iloc[1].split(',')} for _, row in substitute_df.iterrows()]
        
        reassigned_allowed = pd.read_excel(angle_filename_reference, sheet_name='可以重複分配的角度', usecols=[0]).iloc[:, 0].dropna().tolist()
        
        angle_banning_df = pd.read_excel(angle_filename_reference, sheet_name='角度禁止規則', usecols=[0, 1, 2])
        angle_banning_rules = [{"if_appears_in_angle": row.iloc[0].split(','), "banned_angle": row.iloc[1], "banned_angle_logic": row.iloc[2]} for _, row in angle_banning_df.iterrows()]
        
        category_rules_df = pd.read_excel(angle_filename_reference, sheet_name='商品分類及關鍵字條件', usecols=[0, 1, 2])
        category_rules = {row.iloc[0]: {"keywords": row.iloc[1].split(','), "match_all": row.iloc[2]} for _, row in category_rules_df.iterrows()}
        
        features_by_category = load_image_features_with_ivf(train_file)
        original_features_by_category = {k: v.copy() for k, v in features_by_category.items()}

    if (uploaded_zip or input_path) and start_running:
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

        if selected_brand == "ADS":
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
            and not f.startswith('__MACOSX') and not f.startswith('.')
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
    
            for image_file, image_path in image_files:
                if image_file.startswith('.') or os.path.isdir(image_path):
                    continue
                for idx, group in enumerate(group_conditions):
                    if any(substr in image_file for substr in group["set_a"]):
                        group_presence[idx]["set_a_present"] = True
                    if any(substr in image_file for substr in group["set_b"]):
                        group_presence[idx]["set_b_present"] = True
            image_filenames = [img[0] for img in image_files]

            for image_file, image_path in image_files:
                if image_file.startswith('.') or os.path.isdir(image_path):
                    continue
                if any(keyword in image_file for keyword in keywords_to_skip):
                    skipped_images.append({
                        "資料夾": folder, 
                        "圖片": image_file
                    })
                    continue

                skip_image = False
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
                if special_mappings:
                    for substr, mapping in special_mappings.items():
                        if substr in image_file:
                            special_angles = mapping['angles']
                            special_category = mapping['category']
                            category_filename = mapping.get('category_filename')
                            if category_filename:
                                if any(cond in fname for fname in image_filenames for cond in category_filename):
                                    pass
                                else:
                                    special_category = None 
                            if special_category and not folder_special_category:
                                folder_special_category = special_category
                            break

                # 這裡是修改點：加入 try-except，以處理 tif 檔
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
                        num_samples = len(features_by_category[brand][category]["labeled_features"])
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

            filtered_by_category = features_by_category[selected_brand][
                best_category["category"]
            ]["labeled_features"]

            angle_to_number = {
                item["labels"]["angle"]: item["labels"]["number"] 
                for item in filtered_by_category
            }

            used_angles = set()
            final_results = {}
            rule_flags = [False for _ in angle_banning_rules]

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

            for img_data in folder_features:
                image_file = img_data["image_file"]
                special_angles = img_data["special_angles"]
                special_category = img_data["special_category"]
                img_features = img_data["features"]
                if special_angles:
                    valid_special_angles = [
                        angle for angle in special_angles 
                        if angle in angle_to_number
                    ]
                    if valid_special_angles:
                        if len(valid_special_angles) > 1:
                            valid_angles_by_similarity = []
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
                            chosen_angle = None
                            for angle, similarity_percentage in valid_angles_by_similarity:
                                if angle not in reassigned_allowed and angle in used_angles:
                                    pass
                                else:
                                    chosen_angle = angle
                                    best_similarity = similarity_percentage
                                    break
                            
                            if chosen_angle:
                                prefix = angle_to_prefix.get((chosen_angle, best_category["category"]), angle_to_prefix.get((chosen_angle, None), None))
                                # 若基本設定為空值prefix則用folder為prefix
                                cat_setting = category_settings.get(best_category["category"], category_settings.get("其他"))
                                if cat_setting["prefix_mode"] == "single" and cat_setting["prefix"] is None:
                                    prefix = folder

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
                                prefix = angle_to_prefix.get((special_angle, best_category["category"]), angle_to_prefix.get((special_angle, None), None))
                                # 若基本設定為空值prefix則用folder為prefix
                                cat_setting = category_settings.get(best_category["category"], category_settings.get("其他"))
                                if cat_setting["prefix_mode"] == "single" and cat_setting["prefix"] is None:
                                    prefix = folder

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
                            prefix = angle_to_prefix.get((angle, candidate["label"]["category"]), angle_to_prefix.get((angle, None), None))
                            cat_setting = category_settings.get(candidate["label"]["category"], category_settings.get("其他"))
                            if cat_setting["prefix_mode"] == "single" and cat_setting["prefix"] is None:
                                prefix = candidate["folder"]
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
                        prefix = angle_to_prefix.get((angle, candidate["label"]["category"]), angle_to_prefix.get((angle, None), None))
                        cat_setting = category_settings.get(candidate["label"]["category"], category_settings.get("其他"))
                        if cat_setting["prefix_mode"] == "single" and cat_setting["prefix"] is None:
                            prefix = candidate["folder"]
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
                        prefix = angle_to_prefix.get((angle, candidate["label"]["category"]), angle_to_prefix.get((angle, None), None))
                        cat_setting = category_settings.get(candidate["label"]["category"], category_settings.get("其他"))
                        if cat_setting["prefix_mode"] == "single" and cat_setting["prefix"] is None:
                            prefix = candidate["folder"]
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
        'file_uploader_key2': 8,
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
    支援 jpg、jpeg、png、tif。
    """
    return sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff'))]
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
    加載並處理圖片，並加上標示（PNG 或 TIF）（如果需要）。
    使用 lru_cache 進行快取以加速重複讀取。
    """
    # 新增對tif檔的處理：嘗試使用Pillow開啟，失敗則用imagecodecs
    try:
        image = Image.open(image_path).convert('RGB')
    except UnidentifiedImageError:
        with open(image_path, 'rb') as f:
            raw_data = f.read()
            decoded_data = imagecodecs.tiff_decode(raw_data)
            image = Image.fromarray(decoded_data).convert('RGB')

    # 為 PNG 或 TIF 圖片加上標示
    if add_label:
        extension = os.path.splitext(image_path)[1].lower()
        if extension == '.png':
            image = add_png_label(image)
        elif extension in ('.tif', '.tiff'):
            image = add_tif_label(image)

    # 統一圖片大小為 800x800，保留 ImageOps.pad() 的邏輯
    image = ImageOps.pad(image, (800, 800), method=Image.Resampling.LANCZOS)

    return image

def handle_submission(selected_folder, images_to_display, outer_images_to_display, use_full_filename, folder_to_data):
    """
    處理圖片檔名修改的提交邏輯，包含重命名邏輯與重複檢查。
    """
    current_filenames = {}
    temp_filename_changes = {}
    modified_outer_count = 0  # 記錄修改的 outer images 數量
    removed_image_count = 0  # 記錄 images_to_display 被移除的數量

    if not use_full_filename:
        prefix = get_prefix(images_to_display)
        if prefix == "":
            prefix = get_prefix(images_to_display)
    else:
        prefix = ""

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
            default_text = filename_without_ext

        if new_text.strip() == '':
            new_filename = ''
        else:
            if not use_full_filename:
                new_filename = prefix + new_text + extension
            else:
                new_filename = new_text + extension

        current_filenames[image_file] = {'new_filename': new_filename, 'text': new_text}
        temp_filename_changes[image_file] = {'new_filename': new_filename, 'text': new_text}

        if new_filename == '':
            removed_image_count += 1

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
            default_text = filename_without_ext
            if new_text.strip() == '':
                new_filename = ''
            else:
                new_filename = new_text + extension

        if new_text.strip() != default_text:
            current_filenames[outer_image_file] = {'new_filename': new_filename, 'text': new_text}
            temp_filename_changes[outer_image_file] = {'new_filename': new_filename, 'text': new_text}

            if new_filename != '':
                modified_outer_count += 1

    new_filenames = [data['new_filename'] for data in temp_filename_changes.values() if data['new_filename'] != '']
    duplicates = [filename for filename, count in Counter(new_filenames).items() if count > 1]

    if duplicates:
        st.session_state['has_duplicates'] = True
        st.session_state['duplicate_filenames'] = duplicates
        st.session_state['confirmed_changes'][selected_folder] = False
    else:
        st.session_state['has_duplicates'] = False
        st.session_state['confirmed_changes'][selected_folder] = True

        if not use_full_filename:
            sorted_files = sorted(temp_filename_changes.items(), key=lambda x: x[1]['new_filename'])
            rename_counter = 1

            for file, data in sorted_files:
                if data['new_filename'] != '':
                    new_index = str(rename_counter).zfill(2)
                    extension = os.path.splitext(file)[1]
                    new_filename = f"{prefix}{new_index}{extension}"

                    temp_filename_changes[file]['new_filename'] = new_filename
                    temp_filename_changes[file]['text'] = new_index

                    rename_counter += 1

        if selected_folder not in st.session_state['filename_changes']:
            st.session_state['filename_changes'][selected_folder] = {}
        st.session_state['filename_changes'][selected_folder].update(temp_filename_changes)

        for file, data in temp_filename_changes.items():
            text_input_key = f"{selected_folder}_{file}"
            st.session_state[text_input_key] = data['text']

    if num_images_key in st.session_state:
        current_num_images = int(st.session_state[num_images_key])
        st.session_state[num_images_key] = str(max(1, current_num_images - removed_image_count + modified_outer_count))

    ad_images_key = f"{selected_folder}_ad_images"
    ad_images_value = st.session_state.get(ad_images_key)
    data = folder_to_data.get(selected_folder, {})
    data_folder_name = data.get('資料夾', selected_folder)

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

    st.session_state['modified_folders'].add(data_folder_name)

@functools.lru_cache(maxsize=512)
def get_sort_key(image_file):
    filename_changes = st.session_state.get('filename_changes', {}).get(selected_folder, {})
    if image_file in filename_changes:
        new_filename = filename_changes[image_file]['new_filename']
        return new_filename if new_filename else image_file
    return image_file

def add_png_label(image):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except OSError:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 100)

    text = "PNG"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    x = image.width - text_width - 20
    y = 20

    draw.text((x, y), text, font=font, fill="red")

    return image

def add_tif_label(image):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except OSError:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 100)

    text = "TIF"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    x = image.width - text_width - 20
    y = 20

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

                                image_path = os.path.join(img_folder_path, image_file) if image_file in image_files else os.path.join(outer_folder_path, image_file)
                                if image_path not in st.session_state['image_cache'][selected_folder]:
                                    add_label = image_file.lower().endswith('.png') or image_file.lower().endswith('.tif') or image_file.lower().endswith('.tiff')
                                    image = load_and_process_image(image_path, add_label)
                                    st.session_state['image_cache'][selected_folder][image_path] = image
                                else:
                                    image = st.session_state['image_cache'][selected_folder][image_path]

                                col.image(image, use_container_width=True)

                                filename_without_ext = os.path.splitext(image_file)[0]
                                extension = os.path.splitext(image_file)[1]

                                if use_full_filename:
                                    default_text = filename_without_ext
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
                                if text_input_key not in st.session_state:
                                    st.session_state[text_input_key] = modified_text

                                col.text_input('檔名', key=text_input_key, label_visibility="collapsed")

                            colA,colB,colC,colD,colE = st.columns(5)
                            col1, col2, col3 ,col4= st.columns([1.1,1.71,1.23,1.23], vertical_alignment="center")
                            if outer_images_to_display:
                                with col4.popover("查看外層圖片"):
                                    outer_cols = st.columns(6)
                                    for idx, outer_image_file in enumerate(outer_images_to_display):
                                        if idx % 6 == 0 and idx != 0:
                                            outer_cols = st.columns(6)
                                        col = outer_cols[idx % 6]

                                        outer_image_path = os.path.join(outer_folder_path, outer_image_file) if outer_image_file in outer_images else os.path.join(img_folder_path, outer_image_file)

                                        if outer_image_path not in st.session_state['image_cache'][selected_folder]:
                                            add_label = outer_image_file.lower().endswith('.png') or outer_image_file.lower().endswith('.tif') or outer_image_file.lower().endswith('.tiff')
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
                                            if first_underscore_index != -1:
                                                default_text = filename_without_ext[first_underscore_index + 1:]
                                            else:
                                                default_text = filename_without_ext

                                        if (selected_folder in st.session_state['filename_changes'] and
                                            outer_image_file in st.session_state['filename_changes'][selected_folder]):
                                            modified_text = st.session_state['filename_changes'][selected_folder][outer_image_file]['text']
                                            if modified_text == '':
                                                modified_text = st.session_state['filename_changes'][selected_folder][outer_image_file].get('last_non_empty', default_text)
                                        else:
                                            modified_text = default_text

                                        text_input_key = f"outer_{selected_folder}_{outer_image_file}"
                                        col.text_input('檔名', value=modified_text, key=text_input_key)

                            if folder_to_data:
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
                            
                                upper_limit = max(10, int(num_images_default), int(ad_images_default))
                            
                                num_images_options = [str(i) for i in range(1, upper_limit + 1)]
                                ad_images_options = [str(i) for i in range(1, upper_limit + 1)]
                                type_images_options = [str(i) for i in range(0, 11)]
                            
                                colA.selectbox('張數', num_images_options, key=num_images_key)
                                colB.selectbox('廣告圖', ad_images_options, key=ad_images_key)
                            
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
                                        top_level_files = [name for name in os.listdir(tmpdirname) if os.path.isfile(os.path.join(tmpdirname, name))]

                                        for file_name in top_level_files:
                                            file_path = os.path.join(tmpdirname, file_name)
                                            arcname = file_name
                                            try:
                                                if file_name != '編圖結果.xlsx':
                                                    zipf.write(file_path, arcname=arcname)
                                            except Exception as e:
                                                st.error(f"壓縮檔案時發生錯誤：{file_name} - {str(e)}")

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
                                                            if new_rel_path not in zipf.namelist():
                                                                zipf.write(full_path, arcname=new_rel_path)
                                                        except Exception as e:
                                                            st.error(f"壓縮檔案時發生錯誤：{full_path} - {str(e)}")
                                                    else:
                                                        try:
                                                            zipf.write(full_path, arcname=rel_path)
                                                        except Exception as e:
                                                            st.error(f"壓縮檔案時發生錯誤：{full_path} - {str(e)}")

                                        excel_buffer = BytesIO()

                                        if excel_sheets:
                                            result_df = excel_sheets.get('編圖張數與廣告圖', pd.DataFrame(columns=['資料夾', '張數', '廣告圖']))
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

                                            type_result_df = excel_sheets.get('圖片類型統計', pd.DataFrame(columns=['資料夾', '模特', '平拍']))
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
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",".tif"]
    EXCLUDED_EXTENSIONS = [".xlsx", ".gsheet"]  # 不需要刪除或複製的檔案類型

    if 'file_uploader_key3' not in st.session_state:
        st.session_state['file_uploader_key3'] = 16
    if 'text_area_key3' not in st.session_state:
        st.session_state['text_area_key3'] = 16

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
