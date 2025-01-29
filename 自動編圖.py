import streamlit as st
import pandas as pd
import zipfile
import os
import torch
import pickle
import shutil
import numpy as np
import re
import tempfile
import chardet
import faiss  
import imagecodecs
from torchvision import models, transforms
from io import BytesIO
from PIL import Image, UnidentifiedImageError

def tab1():
    def initialize_tab1():
        # 設定所有預設值
        defaults = {
            'file_uploader_key1': 0,
            'text_area_key1': 0,
            'file_uploader_disabled_1': False,
            'text_area_disabled_1': False,
            'text_area_content': "",
            'previous_uploaded_file_name_tab1': None,
            'previous_input_path_tab1': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
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
            last_selected_brand = ""
    
    initialize_tab1()
    resnet = load_resnet_model()
    preprocess = get_preprocess_transforms()

    st.write("\n")
    col1, col2 = st.columns(2)

    uploaded_zip = col1.file_uploader(
        "上傳 ZIP 檔案",
        type=["zip"],
        key='file_uploader_' + str(st.session_state['file_uploader_key1']),
        disabled=st.session_state['file_uploader_disabled_1'],
        on_change=handle_file_uploader_change,
        label_visibility="collapsed"
    )

    input_path = col2.text_area(
        "   輸入資料夾路徑",
        height=72,
        key='text_area_' + str(st.session_state['text_area_key1']),
        disabled=st.session_state['text_area_disabled_1'],
        on_change=handle_text_area_change,
        placeholder = "  輸入分包資料夾路徑",
        label_visibility="collapsed"
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

