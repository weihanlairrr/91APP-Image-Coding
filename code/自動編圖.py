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
from PIL import Image, UnidentifiedImageError, ImageCms
import concurrent.futures
import subprocess
import stat

def tab1():
    # 定義 onerror 處理函式，嘗試移除唯讀屬性後重試刪除
    def on_rm_error(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)
        
    # 定義刪除檔案的輔助函式
    def remove_file(path):
        try:
            os.remove(path)
        except PermissionError:
            os.chmod(path, stat.S_IWRITE)
            os.remove(path)
    
    # 設定 Faiss 執行緒數量
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    faiss.omp_set_num_threads(min(32, (os.cpu_count() or 1) + 4))
    
    # =============================================================================
    # 基本工具與檔案處理函式
    # =============================================================================
    def initialize_tab1():
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
        brand_path = os.path.join("dependencies", brand_name)
        train_file = None
        angle_filename_reference = None
        for filename in os.listdir(brand_path):
            lower_filename = filename.lower()
            if lower_filename.endswith(".pkl") and ("image_features" in lower_filename):
                train_file = os.path.join(brand_path, filename)
            if lower_filename.endswith(".xlsx") and ("檔名角度對照表" in lower_filename):
                angle_filename_reference = os.path.join(brand_path, filename)
        return train_file, angle_filename_reference

    def copytree_multithreaded(src, dst):
        if os.path.exists(dst):
            shutil.rmtree(dst, onerror=on_rm_error)
        os.makedirs(dst, exist_ok=True)
        try:
            if os.name == 'nt':
                cmd = ['robocopy', src, dst, '/MIR', '/MT:20', '/R:0', '/W:0', '/NFL', '/NDL', '/NJH', '/NJS']
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                cmd = ['rsync', '-a', src + '/', dst + '/']
                subprocess.run(cmd)
        except Exception:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                futures = []
                for root, dirs, files in os.walk(src):
                    rel_path = os.path.relpath(root, src)
                    dst_dir = os.path.join(dst, rel_path) if rel_path != '.' else dst
                    os.makedirs(dst_dir, exist_ok=True)
                    for file in files:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(dst_dir, file)
                        futures.append(executor.submit(shutil.copy2, src_file, dst_file))
                concurrent.futures.wait(futures)

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

    # =============================================================================
    # 數學運算、標準化與索引建立函式
    # =============================================================================
    def get_dynamic_nlist(num_samples):
        if num_samples >= 1000:
            return min(200, int(np.sqrt(num_samples)))
        elif num_samples >= 100:
            return min(100, int(np.sqrt(num_samples)))
        else:
            return max(1, num_samples // 2)

    def l2_normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def build_ivf_index(features, nlist):
        d = features.shape[1]
        nlist = min(nlist, len(features))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(features)
        index.add(features)
        return index

    # =============================================================================
    # 模型載入與影像前處理
    # =============================================================================
    @st.cache_resource
    def load_resnet_model():
        device = torch.device("cpu")
        weights_path = "dependencies/resnet50.pt"
        resnet = models.resnet50()
        resnet.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        resnet.eval().to(device)
        return resnet

    def process_category_and_build_index(category, data):
        labeled_feats = data["labeled_features"]
        features = np.array([item["features"] for item in labeled_feats], dtype=np.float32)
        features = l2_normalize(features)
        num_samples = len(features)
        nlist = get_dynamic_nlist(num_samples)
        index = build_ivf_index(features, nlist)
        return category, index

    @st.cache_resource
    def load_image_features_with_ivf(train_file_path):
        with open(train_file_path, 'rb') as f:
            features_by_category = pickle.load(f)
        for brand, categories in features_by_category.items():
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                futures = {}
                for cat, data in categories.items():
                    fut = executor.submit(process_category_and_build_index, cat, data)
                    futures[fut] = cat
                for future in concurrent.futures.as_completed(futures):
                    cat_, idx = future.result()
                    features_by_category[brand][cat_]["index"] = idx
        return features_by_category

    def get_image_features(image, model, preprocess):
        device = torch.device("cpu")
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image).cpu().numpy().flatten()
        return features

    # =============================================================================
    # UI 狀態與事件處理函式
    # =============================================================================
    def handle_file_uploader_change():
        file_key = 'file_uploader_' + str(st.session_state.get('file_uploader_key1', 0))
        uploaded_file_1 = st.session_state.get(file_key, None)
        if uploaded_file_1:
            current_filename = uploaded_file_1.name
            if current_filename != st.session_state.get('previous_uploaded_file_name_tab1', None):
                try:
                    if os.path.exists("uploaded_images"):
                        shutil.rmtree("uploaded_images", onerror=on_rm_error)
                    if "custom_tmpdir" in st.session_state and st.session_state["custom_tmpdir"]:
                        shutil.rmtree(st.session_state["custom_tmpdir"], onerror=on_rm_error)
                except:
                    pass
                st.session_state["custom_tmpdir"] = tempfile.mkdtemp()
                st.session_state['previous_uploaded_file_name_tab1'] = current_filename
        st.session_state.text_area_disabled_1 = bool(uploaded_file_1)

    def handle_text_area_change():
        text_key = 'text_area_' + str(st.session_state.get('text_area_key1', 0))
        text_content = st.session_state.get(text_key, "").strip()
        if text_content != st.session_state.get('previous_input_path_tab1', None):
            try:
                if os.path.exists("uploaded_images"):
                    shutil.rmtree("uploaded_images", onerror=on_rm_error)
                if "custom_tmpdir" in st.session_state and st.session_state["custom_tmpdir"]:
                    shutil.rmtree(st.session_state["custom_tmpdir"], onerror=on_rm_error)
            except:
                pass
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
        st.session_state['file_uploader_key1'] += 1
        st.session_state['text_area_key1'] += 1
        st.session_state['file_uploader_disabled_1'] = False
        st.session_state['text_area_disabled_1'] = False

    # =============================================================================
    # 分類比對與統計函式
    # =============================================================================
    def category_match(image_files, keywords, match_all):
        if match_all:
            return all(any(keyword in image_file for image_file in image_files) for keyword in keywords)
        else:
            return any(any(keyword in image_file for image_file in image_files) for keyword in keywords)

    # 這裡保留 is_banned_angle 函式，但後續候選角度已先行移除被禁止角度
    def is_banned_angle(item_angle, rule_flags):
        for idx, rule in enumerate(angle_banning_rules):
            if rule_flags[idx]:
                if rule["banned_angle_logic"] == "等於":
                    if any(item_angle == banned for banned in rule["banned_angle"]):
                        return True
                elif rule["banned_angle_logic"] == "包含":
                    if any(banned in item_angle for banned in rule["banned_angle"]):
                        return True
        return False

    def generate_image_type_statistics(results):
        filtered_results = results[
            (results["編號"] != "超過可編上限") &
            (results["編號"] != "不編的角度") &
            (results["編號"] != "")
        ]
        statistics = []
        for folder, folder_results in filtered_results.groupby("資料夾"):
            model_count = folder_results["角度"].apply(lambda x: ("模特" in x) or ("_9" in x) or ("-0m" in x)).sum()
            flat_lay_count = len(folder_results) - model_count
            statistics.append({
                "資料夾": folder,
                "模特": model_count,
                "平拍": flat_lay_count,
            })
        return pd.DataFrame(statistics)

    # =============================================================================
    # 影像重新命名與打包相關函式
    # =============================================================================
    def get_prefix(angle, best_category, folder, angle_to_prefix):
        prefix = angle_to_prefix.get((angle, best_category["category"]), angle_to_prefix.get((angle, None), None))
        cat_setting = category_settings.get(best_category["category"], category_settings.get("其他"))
        if cat_setting["prefix_mode"] == "single" and cat_setting["prefix"] is None:
            prefix = folder
        return prefix

    def rename_numbers_in_folder(results, output_excel_data, skipped_images, folder_settings, angle_to_prefix, selected_brand):
        df = pd.DataFrame(results)
        df["前綴"] = df.get("前綴", None)
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
                if prefix is None:
                    prefix = folder
                valid_idx = folder_df[~folder_df["numeric_編號"].isna()].index
                for i, idx_ in enumerate(valid_idx):
                    if i < label_limit:
                        folder_df.at[idx_, "編號"] = f"{start_num + i:02d}"
                    else:
                        folder_df.at[idx_, "編號"] = "超過可編上限"
            else:
                prefix_list = cat_setting["prefixes"]
                label_limits = cat_setting["label_limits"]
                start_numbers = cat_setting["start_numbers"]
                folder_df.loc[~folder_df["前綴"].isin(prefix_list), "編號"] = np.nan
                for p_idx, pfx in enumerate(prefix_list):
                    pfx_limit = label_limits[p_idx]
                    pfx_start = start_numbers[p_idx]
                    subset_idx = folder_df[(folder_df["前綴"] == pfx) & (~folder_df["numeric_編號"].isna())].index
                    for i, idx_ in enumerate(subset_idx):
                        if i < pfx_limit:
                            folder_df.at[idx_, "編號"] = f"{pfx_start + i:02d}"
                        else:
                            folder_df.at[idx_, "編號"] = "超過可編上限"
            folder_df.drop(columns=["numeric_編號"], inplace=True)
            new_results.append(folder_df)
        new_df = pd.concat(new_results, ignore_index=True)
        return new_df.to_dict('records')

    def rename_and_zip_folders(results, output_excel_data, skipped_images, folder_settings, angle_to_prefix, selected_brand):
        output_folder_path = "uploaded_images"
        unmodified_files = {}  
        for result in results:
            folder_name = result["資料夾"]
            image_file = result["圖片"]
            new_number = result["編號"]
            prefix = result.get("前綴", None)
            unmodified_reason = result["不編原因"]
            if prefix is None:
                prefix = folder_name
            folder_path = os.path.join(output_folder_path, folder_name)
            use_two_img_folder = folder_settings.get(folder_name, False)
            main_folder_structure = "2-IMG" if use_two_img_folder else "1-Main/All"
            main_folder_path = os.path.join(folder_path, main_folder_structure)
            os.makedirs(main_folder_path, exist_ok=True)
            old_image_path = os.path.join(folder_path, image_file)
            file_extension = os.path.splitext(image_file)[1].lower()
            
            if unmodified_reason != "":
                if use_two_img_folder:
                    unmodified_files.setdefault(folder_name, []).append(old_image_path)
                else:
                    new_image_path = os.path.join(folder_path, os.path.basename(image_file))
                    if os.path.exists(old_image_path) and os.path.normpath(old_image_path) != os.path.normpath(new_image_path):
                        os.rename(old_image_path, new_image_path)
            else:
                new_image_name = f"{prefix}{new_number}{file_extension}" if use_two_img_folder else f"{prefix}_{new_number}{file_extension}"
                new_image_path = os.path.join(main_folder_path, new_image_name)
                if os.path.exists(old_image_path) and os.path.normpath(old_image_path) != os.path.normpath(new_image_path):
                    os.rename(old_image_path, new_image_path)
                    
        if selected_brand == "TNF":
            for folder_name, file_list in unmodified_files.items():
                use_two_img_folder = folder_settings.get(folder_name, False)
                if not use_two_img_folder:
                    continue
                inner_folder = "2-IMG"
                target_folder = os.path.join(output_folder_path, folder_name, inner_folder)
                file_list_sorted = sorted(file_list, key=lambda x: os.path.basename(x))
                counter = 101
                for old_path in file_list_sorted:
                    ext = os.path.splitext(old_path)[1].lower()
                    new_name = f"{counter}{ext}"
                    new_path = os.path.join(target_folder, new_name)
                    if os.path.exists(old_path):
                        os.rename(old_path, new_path)
                    counter += 1
    
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipf:
            for folder in os.listdir("uploaded_images"):
                folder_path = os.path.join("uploaded_images", folder)
                if not os.path.isdir(folder_path):
                    continue
                use_two_img_folder = folder_settings.get(folder, False)
                new_folder_name = folder if use_two_img_folder else f"{folder}_OK"
                new_folder_path = os.path.join("uploaded_images", new_folder_name)
                os.rename(folder_path, new_folder_path)
                for root, dirs, files in os.walk(new_folder_path):
                    for dir_name in dirs:
                        dir_full_path = os.path.join(root, dir_name)
                        zip_dir_path = os.path.relpath(dir_full_path, "uploaded_images") + "/"
                        if zip_dir_path not in zipf.namelist():
                            zip_info = zipfile.ZipInfo(zip_dir_path)
                            zip_info.external_attr = (0o755 & 0xFFFF) << 16
                            zipf.writestr(zip_info, b"", compress_type=zipfile.ZIP_STORED)
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, "uploaded_images")
                        if os.path.splitext(file)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif']:
                            zipf.write(file_path, arcname, compress_type=zipfile.ZIP_STORED)
                        else:
                            zipf.write(file_path, arcname)
            zipf.writestr(f"{selected_brand}編圖結果.xlsx", output_excel_data)
        return zip_buffer.getvalue()

    # =============================================================================
    # 其他 UI 事件處理
    # =============================================================================
    def update_brand():
        new_brand = st.session_state["brand_selectbox"]
        if new_brand != "":
            with open(selected_brand_file, "w", encoding="utf-8") as f:
                f.write(new_brand)

    # =============================================================================
    # 單一影像處理函式
    # =============================================================================
    def process_single_image(image_file, image_path, group_presence, group_conditions, keywords_to_skip, special_mappings, folder, preprocess, resnet):
        return_dict = {
            "skip": False,
            "image_file": image_file,
            "features": None,
            "special_angles": [],
            "special_category": None
        }
        for keyword in keywords_to_skip:
            if keyword in image_file:
                return_dict["skip"] = True
                return_dict["skip_reason"] = f"檔名含有{keyword}"
                return return_dict
        for idx, group in enumerate(group_conditions):
            if any(substr in image_file for substr in group["set_b"]):
                if group_presence["set_a_present"][idx] and group_presence["set_b_present"][idx]:
                    return_dict["skip"] = True
                    return_dict["skip_reason"] = f"已存在檔名含有{'、'.join(group['set_a'])}的圖片"
                    return return_dict
        special_angles = []
        special_category = None
        category_filename = None
        for substr, mapping in special_mappings.items():
            if substr in image_file:
                special_angles = mapping['angles']
                special_category = mapping['category']
                category_filename = mapping.get('category_filename')
                if category_filename:
                    pass
                break
        return_dict["special_angles"] = special_angles
        return_dict["special_category"] = special_category
        file_size = os.path.getsize(image_path)
        file_extension = os.path.splitext(image_file)[1].lower()
        try:
            if file_size > 5 * 1024 * 1024 and file_extension != ".png":
                large_img = Image.open(image_path)
                icc_profile = large_img.info.get('icc_profile')
                if large_img.mode != "RGB":
                    srgb_profile = ImageCms.createProfile("sRGB")
                    if icc_profile:
                        input_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
                        large_img = ImageCms.profileToProfile(large_img, input_profile, srgb_profile, outputMode="RGB")
                    else:
                        large_img = large_img.convert("RGB")
                new_width = large_img.width // 2
                new_height = large_img.height // 2
                resized_img = large_img.resize((new_width, new_height), Image.LANCZOS)
                if icc_profile:
                    resized_img.save(image_path, icc_profile=icc_profile)
                else:
                    resized_img.save(image_path)
                img = resized_img
            else:
                img = Image.open(image_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
        except UnidentifiedImageError:
            with open(image_path, 'rb') as f:
                raw_data = f.read()
            decoded_data = imagecodecs.tiff_decode(raw_data)
            img = Image.fromarray(decoded_data)
            if img.mode != "RGB":
                img = img.convert("RGB")
        img_feat = get_image_features(img, resnet, preprocess)
        return_dict["features"] = img_feat
        return return_dict

    # =============================================================================
    # 主流程：讀取設定、檔案處理、分類比對、影像重新命名與打包下載
    # =============================================================================
    selected_brand_file = "dependencies/selected_brand.txt"
    brand_folders = [f for f in os.listdir("dependencies") if os.path.isdir(os.path.join("dependencies", f))
                     and not f.startswith('.') and f != "__pycache__"]
    brand_list = brand_folders
    if os.path.exists(selected_brand_file):
        with open(selected_brand_file, "r", encoding="utf-8") as f:
            last_selected_brand = f.read().strip()
        if last_selected_brand not in brand_list:
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
    st.write("\n")
    col1, col2 = st.columns(2, vertical_alignment="top")
    uploaded_zip = col1.file_uploader(
        "上傳 ZIP 檔案",
        type=["zip"],
        key='file_uploader_' + str(st.session_state['file_uploader_key1']),
        disabled=st.session_state['file_uploader_disabled_1'],
        on_change=handle_file_uploader_change,
        label_visibility="collapsed"
    )
    input_path = col2.text_area(
        "輸入資料夾路徑",
        height=68,
        key='text_area_' + str(st.session_state['text_area_key1']),
        disabled=st.session_state['text_area_disabled_1'],
        on_change=handle_text_area_change,
        placeholder="  輸入分包資料夾路徑",
        label_visibility="collapsed"
    )
    start_running = False
    if input_path:
        st.session_state["input_path_from_tab1"] = input_path
    if uploaded_zip or input_path:
        col1_, col2_, col3_ = st.columns([1.5, 2, 5], vertical_alignment="center", gap="medium")
        selectbox_placeholder = col1_.empty()
        button_placeholder = col2_.empty()
        with selectbox_placeholder:
            if brand_list:
                selected_brand_index = brand_list.index(last_selected_brand) if last_selected_brand in brand_list else 0
                selected_brand = st.selectbox(
                    "請選擇品牌",
                    brand_list,
                    index=selected_brand_index,
                    label_visibility="collapsed",
                    key="brand_selectbox",
                    on_change=update_brand
                )
            else:
                selected_brand = st.selectbox("請選擇品牌", [], label_visibility="collapsed", key="brand_selectbox", on_change=update_brand)
        with button_placeholder:
            start_running = st.button("開始執行")
        if (uploaded_zip or input_path) and start_running:
            selectbox_placeholder.empty()
            button_placeholder.empty()
            if not selected_brand:
                st.error("未偵測到任何品牌資料夾，請確認 'dependencies' 下是否有子資料夾。")
                st.stop()
            if selected_brand == "STB":
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, 10.0)),
                    transforms.Normalize(mean=[0.485, 0.44, 0.406], std=[0.2, 0.2, 0.2]),
                ])
            else:
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.44, 0.406], std=[0.2, 0.2, 0.2]),
                ])
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
                category_settings["其他"] = {"prefix_mode": "single", "prefix": None, "label_limit": 3, "start_number": 1}
            df_skip = pd.read_excel(
                angle_filename_reference,
                sheet_name='不編的檔名或角度',
                usecols=[0, 1]
            )
            keywords_to_skip = df_skip.iloc[:, 0].dropna().astype(str).tolist()
            angle_keywords_to_skip = df_skip.iloc[:, 1].dropna().astype(str).tolist()
            substitute_df = pd.read_excel(
                angle_filename_reference,
                sheet_name='有條件使用的檔名',
                usecols=[0, 1]
            )
            substitute = []
            for _, row in substitute_df.iterrows():
                set_a = row.iloc[0]
                set_b = row.iloc[1]
                if pd.isna(set_a) or pd.isna(set_b):
                    continue
                substitute.append({"set_a": set_a.split(','), "set_b": set_b.split(',')})
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
            angle_banning_rules = []
            for _, row in angle_banning_df.iterrows():
                if_appears_in_angle = str(row.iloc[0]).split(',')
                banned_angle = str(row.iloc[1]).split(',')
                banned_logic = str(row.iloc[2]).strip()
                angle_banning_rules.append({
                    "if_appears_in_angle": if_appears_in_angle,
                    "banned_angle": banned_angle,
                    "banned_angle_logic": banned_logic
                })
            category_rules_df = pd.read_excel(
                angle_filename_reference,
                sheet_name='商品分類及關鍵字條件',
                usecols=[0, 1, 2]
            )
            category_rules = {}
            for _, row in category_rules_df.iterrows():
                cat = row.iloc[0]
                if pd.isna(cat):
                    continue
                keywords = row.iloc[1].split(',')
                match_all = row.iloc[2]
                category_rules[cat] = {"keywords": keywords, "match_all": match_all}
            features_by_category = load_image_features_with_ivf(train_file)
            original_features_by_category = {k: v.copy() for k, v in features_by_category.items()}

            if os.path.exists("uploaded_images") and os.path.isdir("uploaded_images"):
                shutil.rmtree("uploaded_images", onerror=on_rm_error)
            if os.path.exists("temp.zip") and os.path.isfile("temp.zip"):
                remove_file("temp.zip")
            with st.spinner("   讀取檔案中，請稍候..."):
                if uploaded_zip:
                    with open("temp.zip", "wb") as f:
                        f.write(uploaded_zip.getbuffer())
                    unzip_file("temp.zip")
                elif input_path:
                    if not os.path.exists(input_path):
                        st.error("指定的本地路徑不存在，請重新輸入。")
                        st.stop()
                    else:
                        copytree_multithreaded(input_path, "uploaded_images")
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
                and f != "0-上架資料"
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
                    if folder != "0-上架資料":
                        st.warning(f"資料夾 {folder} 中沒有有效的圖片，跳過此資料夾")
                    shutil.rmtree(folder_path, onerror=on_rm_error)
                    processed_folders += 1
                    progress_bar.progress(processed_folders / total_folders)
                    continue
                progress_text.text(f"正在處理資料夾: {folder}")
                image_filenames = [img[0] for img in image_files]
                local_group_presence = {"set_a_present": [], "set_b_present": []}
                for _ in group_conditions:
                    local_group_presence["set_a_present"].append(False)
                    local_group_presence["set_b_present"].append(False)
                for image_file in image_filenames:
                    for idx, group in enumerate(group_conditions):
                        if any(substr in image_file for substr in group["set_a"]):
                            local_group_presence["set_a_present"][idx] = True
                        if any(substr in image_file for substr in group["set_b"]):
                            local_group_presence["set_b_present"][idx] = True

                folder_features = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                    futures = []
                    for image_file, image_path in image_files:
                        future = executor.submit(
                            process_single_image,
                            image_file, image_path,
                            local_group_presence, group_conditions,
                            keywords_to_skip, special_mappings,
                            folder, preprocess, resnet
                        )
                        futures.append(future)
                    for fut in concurrent.futures.as_completed(futures):
                        res_dict = fut.result()
                        if res_dict.get("skip", False):
                            skipped_images.append({
                                "資料夾": folder,
                                "圖片": res_dict["image_file"],
                                "skip_reason": res_dict.get("skip_reason", "已無可分配的角度"),
                                "skip_record": True
                            })
                        else:
                            folder_features.append({
                                "image_file": res_dict["image_file"],
                                "features": res_dict["features"],
                                "special_angles": res_dict["special_angles"],
                                "special_category": res_dict["special_category"]
                            })

                folder_special_category = None
                for item in folder_features:
                    if item["special_category"]:
                        folder_special_category = item["special_category"]
                        break

                if folder_special_category is None:
                    for category, rule in list(category_rules.items()):
                        if category in features_by_category.get(selected_brand, {}):
                            if not category_match(image_filenames, rule["keywords"], rule["match_all"]):
                                features_by_category[selected_brand].pop(category, None)
                if len(folder_features) == 0:
                    if folder != "0-上架資料":
                        st.warning(f"資料夾 {folder} 中沒有有效的圖片，跳過此資料夾")
                    shutil.rmtree(folder_path, onerror=on_rm_error)
                    processed_folders += 1
                    progress_bar.progress(processed_folders / total_folders)
                    continue
                if folder_special_category:
                    best_category = {'brand': selected_brand, 'category': folder_special_category}
                else:
                    category_similarities = {}
                    if selected_brand not in features_by_category:
                        st.warning(f"品牌 {selected_brand} 不存在於特徵字典中，跳過資料夾 {folder}")
                        processed_folders += 1
                        progress_bar.progress(processed_folders / total_folders)
                        continue
                    for category in features_by_category[selected_brand]:
                        index = features_by_category[selected_brand][category]["index"]
                        nlist = index.nlist
                        nprobe = max(1, int(np.sqrt(nlist)))
                        index.nprobe = nprobe
                        folder_similarities = []
                        for img_data in folder_features:
                            img_feat = img_data["features"].astype(np.float32).reshape(1, -1)
                            img_feat = l2_normalize(img_feat)
                            sims, _ = index.search(img_feat, k=5)
                            sims = sims.flatten()
                            nonzero_sims = sims[sims != 0]
                            if len(nonzero_sims) > 0:
                                avg_similarity = np.mean(nonzero_sims)
                            else:
                                avg_similarity = 0
                            folder_similarities.append(avg_similarity)
                        category_similarities[category] = np.mean(folder_similarities)
                    if category_similarities:
                        best_category_name = max(category_similarities, key=category_similarities.get)
                        best_category = {'brand': selected_brand, 'category': best_category_name}
                    else:
                        st.warning(f"資料夾 {folder} 無法匹配任何分類，跳過此資料夾")
                        shutil.rmtree(folder_path, onerror=on_rm_error)
                        processed_folders += 1
                        progress_bar.progress(processed_folders / total_folders)
                        continue

                # 建立候選角度清單
                filtered_by_category = features_by_category[selected_brand][best_category["category"]]["labeled_features"]
                angle_to_number = { item["labels"]["angle"]: item["labels"]["number"] for item in filtered_by_category }
                used_angles = set()
                final_results = {}

                angle_feats_map = {}
                for item in filtered_by_category:
                    ang = item["labels"]["angle"]
                    if ang not in angle_feats_map:
                        angle_feats_map[ang] = []
                    angle_feats_map[ang].append(item["features"])
                angle_index = {}
                for ang, feats_list in angle_feats_map.items():
                    feats_arr = np.array(feats_list, dtype=np.float32)
                    feats_arr = l2_normalize(feats_arr)
                    idx = faiss.IndexFlatIP(feats_arr.shape[1])
                    idx.add(feats_arr)
                    angle_index[ang] = idx

                # ──【重點修改】在此依據資料夾中所有圖片檔名，檢查是否觸發任一角度禁止規則，
                # 若觸發，則根據「等於」或「包含」的設定，從候選角度清單中移除該禁止角度。
                folder_triggered_rules = [False] * len(angle_banning_rules)
                for idx, rule in enumerate(angle_banning_rules):
                    for img_fname in image_filenames:
                        for trigger in rule["if_appears_in_angle"]:
                            if trigger in img_fname:
                                folder_triggered_rules[idx] = True
                                break
                        if folder_triggered_rules[idx]:
                            break
                # 將觸發的規則對應的禁止角度從候選清單中移除
                for idx, triggered in enumerate(folder_triggered_rules):
                    if triggered:
                        for banned_ang in angle_banning_rules[idx]["banned_angle"]:
                            if angle_banning_rules[idx]["banned_angle_logic"] == "等於":
                                if banned_ang in angle_to_number:
                                    angle_to_number.pop(banned_ang)
                                if banned_ang in angle_index:
                                    angle_index.pop(banned_ang)
                                if banned_ang in angle_feats_map:
                                    angle_feats_map.pop(banned_ang)
                            elif angle_banning_rules[idx]["banned_angle_logic"] == "包含":
                                for candidate in list(angle_to_number.keys()):
                                    if banned_ang in candidate:
                                        angle_to_number.pop(candidate, None)
                                        if candidate in angle_index:
                                            angle_index.pop(candidate)
                                        if candidate in angle_feats_map:
                                            angle_feats_map.pop(candidate)
                # 將 rule_flags 設為 folder_triggered_rules 供後續 is_banned_angle 使用（理論上候選角度已移除）
                rule_flags = folder_triggered_rules
                # ──【修改結束】

                # 處理 special_angles 的圖片
                for img_data in folder_features:
                    image_file = img_data["image_file"]
                    special_angles = img_data["special_angles"]
                    if image_file in final_results:
                        continue
                    if special_angles:
                        valid_special_angles = [a for a in special_angles if a in angle_to_number]
                        if valid_special_angles:
                            if len(valid_special_angles) > 1:
                                best_angle = None
                                best_similarity = -1
                                for sa in valid_special_angles:
                                    # 由於候選角度中被禁止的已移除，此處直接比對相似度
                                    if sa in used_angles and sa not in reassigned_allowed:
                                        continue
                                    if sa in angle_index:
                                        temp_index = angle_index[sa]
                                        img_query = l2_normalize(img_data["features"].astype(np.float32).reshape(1, -1))
                                        sims, _ = temp_index.search(img_query, k=1)
                                        sim_percent = sims[0][0] * 100
                                        if sim_percent > best_similarity:
                                            best_similarity = sim_percent
                                            best_angle = sa
                                if best_angle:
                                    prefix_ = get_prefix(best_angle, best_category, folder, angle_to_prefix)
                                    used_angles.add(best_angle)
                                    final_results[image_file] = {
                                        "資料夾": folder,
                                        "圖片": image_file,
                                        "商品分類": best_category["category"],
                                        "角度": best_angle,
                                        "編號": angle_to_number[best_angle],
                                        "最大相似度": f"{best_similarity:.2f}%",
                                        "前綴": prefix_
                                    }
                                else:
                                    img_data["special_angles"] = []
                            else:
                                sa = valid_special_angles[0]
                                if sa not in reassigned_allowed and sa in used_angles:
                                    img_data["special_angles"] = []
                                else:
                                    temp_index = angle_index.get(sa)
                                    if temp_index is not None:
                                        img_query = l2_normalize(img_data["features"].astype(np.float32).reshape(1, -1))
                                        prefix_ = get_prefix(sa, best_category, folder, angle_to_prefix)
                                        used_angles.add(sa)
                                        final_results[image_file] = {
                                            "資料夾": folder,
                                            "圖片": image_file,
                                            "商品分類": best_category["category"],
                                            "角度": sa,
                                            "編號": angle_to_number[sa],
                                            "最大相似度": "100.00%",
                                            "前綴": prefix_
                                        }
                                    else:
                                        img_data["special_angles"] = []
                        else:
                            img_data["special_angles"] = []
                # 處理非 special_angles 的圖片
                non_special_images = [x for x in folder_features if not x["special_angles"]]
                if not special_mappings:
                    non_special_images = folder_features
                index = features_by_category[selected_brand][best_category["category"]]["index"]
                nlist = index.nlist
                nprobe = max(1, int(np.sqrt(nlist)))
                index.nprobe = nprobe
                labels = [item["labels"] for item in filtered_by_category]
                cat_setting = category_settings.get(best_category["category"], category_settings.get("其他"))
                if cat_setting["prefix_mode"] == "single":
                    label_limit = cat_setting["label_limit"]
                else:
                    label_limit = cat_setting["label_limits"][0]
                image_similarity_store = {}
                for img_data in non_special_images:
                    image_file = img_data["image_file"]
                    if image_file in final_results:
                        continue
                    img_features_ = img_data["features"].astype(np.float32).reshape(1, -1)
                    img_features_ = l2_normalize(img_features_)
                    similarities, indices = index.search(img_features_, k=len(labels))
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
                        angle_ = candidate["label"]["angle"]
                        if angle_ not in seen_angles:
                            unique_labels.append(candidate)
                            seen_angles.add(angle_)
                        if len(unique_labels) == label_limit:
                            break
                    image_similarity_store[image_file] = unique_labels
                unassigned_images = set(image_similarity_store.keys())
                while unassigned_images:
                    angle_to_images = {}
                    image_current_choices = {}
                    for image_file in unassigned_images:
                        similarity_list = image_similarity_store.get(image_file, [])
                        candidate = None
                        for candidate_candidate in similarity_list:
                            candidate_angle = candidate_candidate["label"]["angle"]
                            if not is_banned_angle(candidate_angle, rule_flags):
                                if candidate_angle not in used_angles or candidate_angle in reassigned_allowed:
                                    candidate = candidate_candidate
                                    break
                        if candidate:
                            angle_ = candidate["label"]["angle"]
                            if angle_ not in angle_to_images:
                                angle_to_images[angle_] = []
                            angle_to_images[angle_].append(image_file)
                            image_current_choices[image_file] = candidate
                    assigned_in_this_round = set()
                    for angle_, images_ in angle_to_images.items():
                        if not images_:
                            continue
                        if angle_ in reassigned_allowed:
                            for imf in images_:
                                cand = image_current_choices.get(imf)
                                if not cand:
                                    continue
                                prefix_ = get_prefix(angle_, cand["label"], cand["folder"], angle_to_prefix)
                                final_results[imf] = {
                                    "資料夾": cand["folder"],
                                    "圖片": imf,
                                    "商品分類": cand["label"]["category"],
                                    "角度": angle_,
                                    "編號": cand["label"]["number"],
                                    "最大相似度": f"{cand['similarity']:.2f}%",
                                    "前綴": prefix_
                                }
                                assigned_in_this_round.add(imf)
                        else:
                            if len(images_) == 1:
                                imf = images_[0]
                                cand = image_current_choices.get(imf)
                                if not cand:
                                    continue
                                prefix_ = get_prefix(angle_, cand["label"], cand["folder"], angle_to_prefix)
                                final_results[imf] = {
                                    "資料夾": cand["folder"],
                                    "圖片": imf,
                                    "商品分類": cand["label"]["category"],
                                    "角度": angle_,
                                    "編號": cand["label"]["number"],
                                    "最大相似度": f"{cand['similarity']:.2f}%",
                                    "前綴": prefix_
                                }
                                used_angles.add(angle_)
                                assigned_in_this_round.add(imf)
                            else:
                                max_sim = -np.inf
                                best_img = None
                                for imf in images_:
                                    cand = image_current_choices.get(imf)
                                    if cand and cand["similarity"] > max_sim:
                                        max_sim = cand["similarity"]
                                        best_img = imf
                                if best_img is not None:
                                    cand = image_current_choices.get(best_img)
                                    prefix_ = get_prefix(angle_, cand["label"], cand["folder"], angle_to_prefix)
                                    final_results[best_img] = {
                                        "資料夾": cand["folder"],
                                        "圖片": best_img,
                                        "商品分類": cand["label"]["category"],
                                        "角度": angle_,
                                        "編號": cand["label"]["number"],
                                        "最大相似度": f"{cand['similarity']:.2f}%",
                                        "前綴": prefix_
                                    }
                                    used_angles.add(angle_)
                                    assigned_in_this_round.add(best_img)
                    unassigned_images -= assigned_in_this_round
                    if not assigned_in_this_round:
                        break
                for image_file in unassigned_images:
                    final_results[image_file] = {
                        "資料夾": folder,
                        "圖片": image_file,
                        "商品分類": best_category["category"],
                        "角度": "已無可分配的角度"
                    }
                for skip in [s for s in skipped_images if s["資料夾"] == folder]:
                    final_results[skip["圖片"]] = {
                        "資料夾": folder,
                        "圖片": skip["圖片"],
                        "商品分類": best_category["category"],
                        "角度": skip["skip_reason"],
                        "編號": "",
                        "前綴": "",
                        "最大相似度": ""
                    }
                for image_file, assignment in final_results.items():
                    if assignment is not None and "角度" in assignment:
                        for skip_keyword in angle_keywords_to_skip:
                            if skip_keyword in assignment["角度"]:
                                assignment["編號"] = "不編的角度"
                                break
                for image_file, assignment in list(final_results.items()):
                    if assignment is not None:
                        results.append(assignment)
                    else:
                        old_image_path = os.path.join(folder_path, image_file)
                        new_image_path = os.path.join("uploaded_images", folder, os.path.basename(image_file))
                        if os.path.exists(old_image_path) and os.path.normpath(old_image_path) != os.path.normpath(new_image_path):
                            os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                            os.rename(old_image_path, new_image_path)
                processed_folders += 1
                progress_bar.progress(processed_folders / total_folders)
            progress_bar.empty()
            progress_text.empty()
            results = rename_numbers_in_folder(results, None, skipped_images, folder_settings, angle_to_prefix, selected_brand)
            for res in results:
                for key in res:
                    if res[key] is None:
                        res[key] = ""
                if res.get("編號", "") in ["超過可編上限", "不編的角度"]:
                    res["不編原因"] = res["編號"]
                    res["編號"] = ""
                if res.get("角度", "") == "已無可分配的角度" or ("檔名" in res.get("角度", "")):
                    res["不編原因"] = res["角度"]
                    res["角度"] = ""
                if "不編原因" not in res:
                    res["不編原因"] = ""
            result_df = pd.DataFrame(results).fillna("")
            desired_order = ["資料夾", "圖片", "商品分類", "角度", "前綴", "編號", "最大相似度", "不編原因"]
            result_df = result_df[desired_order]
            result_df["前綴"] = [
                (row["前綴"] + "_") if (not folder_settings.get(row["資料夾"], False) and not row["不編原因"]) else row["前綴"] 
                for _, row in result_df.iterrows()
            ]

            st.dataframe(result_df, hide_index=True, use_container_width=True, height=457)
            folder_data = []
            for folder in image_folders:
                folder_results = result_df[result_df['資料夾'] == folder]
                valid_images = folder_results[
                    (folder_results['編號'] != '超過可編上限') &
                    (folder_results['編號'] != '不編的角度') &
                    (folder_results['編號'] != '')
                ]
                num_images = len(valid_images)
                ad_images = valid_images[valid_images['角度'].str.contains('情境', na=False)]
                num_ad_images = len(ad_images)
                ad_image_value = f"{num_ad_images + 1:02}" if num_ad_images > 0 else "01"
                folder_data.append({'資料夾': folder, '張數': num_images, '廣告圖': ad_image_value})
            folder_df = pd.DataFrame(folder_data)
            image_type_statistics_df = generate_image_type_statistics(result_df)
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, sheet_name='編圖紀錄', index=False)
                folder_df.to_excel(writer, sheet_name='編圖張數與廣告圖', index=False)
                image_type_statistics_df.to_excel(writer, sheet_name='圖片類型統計', index=False)
            excel_data = excel_buffer.getvalue()
            zip_data = rename_and_zip_folders(results, excel_data, skipped_images, folder_settings, angle_to_prefix, selected_brand)
            if uploaded_zip:
                uploaded_zip_name = os.path.splitext(uploaded_zip.name)[0]
                download_file_name = f"{uploaded_zip_name}_結果.zip"
            elif input_path:
                last_folder_name = os.path.basename(os.path.normpath(input_path))
                download_file_name = f"{last_folder_name}_結果.zip"
            else:
                download_file_name = "結果.zip"
            shutil.rmtree("uploaded_images", onerror=on_rm_error)
            if uploaded_zip:
                remove_file("temp.zip")
            if st.download_button(
                label="下載編圖結果",
                data=zip_data,
                file_name=download_file_name,
                mime="application/zip",
                on_click=reset_key_tab1
            ):
                st.rerun()

