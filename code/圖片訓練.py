import streamlit as st
import pandas as pd
import zipfile
import os
import shutil
import re
import tempfile
import functools
import subprocess
import stat
import concurrent.futures
import fitz
import torch
import pickle
from io import BytesIO
from psd_tools import PSDImage
from PIL import Image, ImageOps, ImageDraw, ImageFont, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

def tab3():
    """
    此函式包含處理編圖結果與圖片特徵萃取的完整流程，
    包含檔案上傳、Excel 檔讀取、圖片載入、角度選擇、以及 ResNet 特徵萃取與資料儲存。
    """

    # =============================================================================
    # 協助函式 - 錯誤處理與初始化
    # =============================================================================
    def on_rm_error(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def initialize_tab3(mode="default"):
        """
        初始化與重置 session_state 和暫存資料夾。
        每次上傳檔案或輸入資料夾路徑時，
        會清除前一次的 selectbox 狀態紀錄（包含 angle 與商品分類）。
        """
        if mode in ("clear_cache", "both"):
            if os.path.exists(fixed_cache_base_dir):
                shutil.rmtree(fixed_cache_base_dir, ignore_errors=False, onerror=on_rm_error)
            os.makedirs(fixed_cache_base_dir, exist_ok=True)
            os.makedirs(fixed_psd_cache_dir, exist_ok=True)
            os.makedirs(fixed_ai_cache_dir, exist_ok=True)
            os.makedirs(fixed_custom_tmpdir, exist_ok=True)
        if mode in ("reinitialize", "both"):
            keys = [
                'image_cache', 'file_uploader_key3', 'text_area_key3', 'modified_folders',
                'previous_input_path', 'file_uploader_disabled_3', 'text_area_disabled_3',
                "custom_tmpdir", 'previous_selected_folder', 'source_loaded',
                'angle_changes', 'category_selection'
            ]
            for key in keys:
                if key in st.session_state:
                    del st.session_state[key]
            for key in list(st.session_state.keys()):
                if key.startswith("prev_"):
                    del st.session_state[key]
        defaults = {
            'image_cache': {},
            'file_uploader_key3': 60,
            'text_area_key3': 60,
            'modified_folders': set(),
            'previous_input_path': None,
            'file_uploader_disabled_3': False,
            'text_area_disabled_3': False,
            "custom_tmpdir": fixed_custom_tmpdir,
            'previous_selected_folder': None,
            'source_loaded': False,
        }
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)
        # 已清除所有 selectbox 狀態，包括 angle_changes 與 category_selection

    # =============================================================================
    # 協助函式 - 多執行緒複製目錄
    # =============================================================================
    def copytree_multithreaded(src, dst):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst, exist_ok=True)
        try:
            if os.name == 'nt':
                cmd = ['robocopy', src, dst, '/MIR', '/MT:20', '/R:0', '/W:0', '/NFL', '/NDL', '/NJH', '/NJS']
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                cmd = ['rsync', '-a', src + '/', dst + '/']
                subprocess.run(cmd)
        except Exception:
            with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
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

    # =============================================================================
    # 協助函式 - 上傳與文字框變更處理
    # =============================================================================
    def handle_file_uploader_change_tab3():
        initialize_tab3(mode="both")
        st.session_state["custom_tmpdir"] = fixed_custom_tmpdir
        file_key = 'file_uploader_' + str(st.session_state.get('file_uploader_key3', 0))
        uploaded_file_1 = st.session_state.get(file_key, None)
        if uploaded_file_1:
            st.session_state['image_cache'].clear()
            st.session_state["source_loaded"] = False
        st.session_state.text_area_disabled_3 = bool(uploaded_file_1)

    def handle_text_area_change_tab3():
        initialize_tab3(mode="both")
        st.session_state["custom_tmpdir"] = fixed_custom_tmpdir
        text_key = 'text_area_' + str(st.session_state.get('text_area_key3', 0))
        text_content = st.session_state.get(text_key, "").strip()
        st.session_state['image_cache'].clear()
        st.session_state['previous_input_path'] = text_content
        st.session_state["source_loaded"] = False
        st.session_state.file_uploader_disabled_3 = bool(text_content)

    # =============================================================================
    # 協助函式 - 圖片處理與快取（立即載入所有圖片）
    # =============================================================================
    def add_image_label(image, file_extension):
        """
        根據檔案副檔名在圖片上加上標籤。
        """
        draw = ImageDraw.Draw(image)
        label_map = {'.png': 'PNG', '.tif': 'TIF', '.tiff': 'TIF', '.psd': 'PSD', '.ai': 'AI'}
        label_text = label_map.get(file_extension.lower())
        if not label_text:
            return image
        font_size = max(30, int(image.width * 0.12))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x = image.width - text_width - 20
        y = 20
        draw.text((x, y), label_text, font=font, fill="red")
        return image

    @functools.lru_cache(maxsize=256)
    def load_and_process_image(image_path, add_label=False):
        """
        載入並處理圖片，依副檔名做不同處理，並回傳 1000x1000 的圖片。
        """
        psd_cache = fixed_psd_cache_dir
        ai_cache = fixed_ai_cache_dir
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.psd':
            os.makedirs(psd_cache, exist_ok=True)
            cache_file_name = str(abs(hash(image_path))) + ".jpg"
            cache_path = os.path.join(psd_cache, cache_file_name)
            if os.path.exists(cache_path):
                image = Image.open(cache_path).convert('RGB')
            else:
                psd = PSDImage.open(image_path, lazy=True)
                image = psd.composite(force=False)
                if image:
                    image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
                    image = image.convert('RGB')
                    image.save(cache_path, format='JPEG', quality=80)
                else:
                    raise Exception("無法處理 PSD 文件")
        elif ext == '.ai':
            os.makedirs(ai_cache, exist_ok=True)
            cache_file_name = str(abs(hash(image_path))) + ".png"
            cache_path = os.path.join(ai_cache, cache_file_name)
            if os.path.exists(cache_path):
                image = Image.open(cache_path)
            else:
                try:
                    doc = fitz.open(image_path)
                    page = doc.load_page(0)
                    pix = page.get_pixmap(dpi=100)
                    image = Image.open(BytesIO(pix.tobytes("png")))
                    doc.close()
                    image.save(cache_path, format='PNG')
                except Exception as e:
                    raise Exception(f"無法處理 .ai 檔案: {str(e)}")
        else:
            # 一般圖片
            image = Image.open(image_path)
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGBA')
            pad_color = (255, 255, 255, 0)
        else:
            image = image.convert('RGB')
            pad_color = (255, 255, 255)
        image = ImageOps.pad(image, (1000, 1000), method=Image.Resampling.LANCZOS, color=pad_color)
        if add_label:
            image = add_image_label(image, ext)
        return image

    def get_outer_folder_images(folder_path):
        """
        取得指定資料夾中符合圖片格式的檔案，並排序後回傳。
        """
        return sorted(
            [f for f in os.listdir(folder_path)
             if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'psd', 'ai'))]
        )

    # =============================================================================
    # 協助函式 - Excel 及其他處理
    # =============================================================================
    def fix_code(val):
        if pd.isna(val):
            return ""
        return str(val).strip()

    def setup_temporary_directory(base_path, tmp_dir, read_folder):
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        for root, dirs, files in os.walk(base_path):
            if tmp_dir in root:
                continue
            is_top_level = os.path.basename(root) in ['1-Main', '2-IMG']
            is_same_level_as_image_folders = os.path.dirname(root) == base_path
            for item in dirs:
                item_path = os.path.join(root, item)
                relative_path = os.path.relpath(item_path, base_path)
                if not (is_same_level_as_image_folders and is_top_level):
                    dest_path = os.path.join(tmp_dir, relative_path)
                    os.makedirs(dest_path, exist_ok=True)
            for item in files:
                item_path = os.path.join(root, item)
                relative_path = os.path.relpath(item_path, base_path)
                ext = os.path.splitext(item)[1].lower()
                if is_same_level_as_image_folders and ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.psd', ".ai"]:
                    continue
                dest_path = os.path.join(tmp_dir, relative_path)
                try:
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(item_path, dest_path)
                except FileNotFoundError as e:
                    st.warning(f"無法建立路徑：{dest_path}，錯誤：{str(e)}")
        for folder_name in os.listdir(tmp_dir):
            folder_path = os.path.join(tmp_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            two_img_path = os.path.join(folder_path, '2-IMG')
            if os.path.exists(two_img_path):
                shutil.rmtree(two_img_path)
            elif os.path.exists(os.path.join(folder_path, '1-Main')):
                shutil.rmtree(os.path.join(folder_path, '1-Main'))

    def handle_submit_angles(folder, angle_keys):
        """
        將使用者的角度選擇寫入 session_state，
        並清除該資料夾所有選項的 widget 狀態。
        """
        angle_selections = {}
        for key, img_rel in angle_keys:
            angle_val = st.session_state.get(key, "")
            angle_selections[img_rel] = angle_val

        if "angle_changes" not in st.session_state:
            st.session_state["angle_changes"] = {}
        st.session_state["angle_changes"][folder] = angle_selections

        for key, _ in angle_keys:
            if key in st.session_state:
                del st.session_state[key]

        st.toast(f"資料夾 {folder} 暫存修改成功!", icon='🎉')

    def get_image_features(image, model):
        """
        使用 ResNet 模型與預處理，萃取圖片特徵。
        """
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model(image_tensor).detach().cpu().numpy().flatten()
        return feats

    def load_existing_features(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def remove_duplicates(features_by_category_):
        """
        移除 features_by_category_ 中重複的 labeled_features，
        並回傳更新後的資料與刪除筆數。
        """
        deleted_count_ = 0
        for b_, categories_ in features_by_category_.items():
            for cat_key, data_ in categories_.items():
                unique_entries = set()
                unique_labeled_features = []
                for item_ in data_["labeled_features"]:
                    features_tuple = tuple(item_["features"])
                    label_tuple = (
                        item_["labels"]["brand"],
                        item_["labels"]["category"],
                        item_["labels"]["angle"],
                        item_["labels"]["number"]
                    )
                    key_ = (features_tuple, label_tuple)
                    if key_ not in unique_entries:
                        unique_entries.add(key_)
                        unique_labeled_features.append(item_)
                    else:
                        deleted_count_ += 1
                features_by_category_[b_][cat_key]["labeled_features"] = unique_labeled_features
        return features_by_category_, deleted_count_

    def reset_tab3():
        st.session_state['file_uploader_key3'] += 1
        st.session_state['text_area_key3'] += 1
        st.session_state['file_uploader_disabled_3'] = False
        st.session_state['text_area_disabled_3'] = False
        initialize_tab3()

    # =============================================================================
    # 全域變數 - 暫存資料夾路徑設定
    # =============================================================================
    fixed_cache_base_dir = os.path.join(tempfile.gettempdir(), "streamlit_cache")
    fixed_psd_cache_dir = os.path.join(fixed_cache_base_dir, "psd_cache")
    fixed_ai_cache_dir = os.path.join(fixed_cache_base_dir, "ai_cache")
    fixed_custom_tmpdir = os.path.join(fixed_cache_base_dir, "custom_tmpdir")

    # =============================================================================
    # 介面初始化與檔案上傳/資料夾路徑輸入
    # =============================================================================
    initialize_tab3()
    st.write("\n")
    col1, col2 = st.columns(2, vertical_alignment="top")
    uploaded_file_3 = col1.file_uploader(
        "上傳編圖結果 ZIP 檔",
        type=["zip"],
        key='file_uploader_' + str(st.session_state['file_uploader_key3']),
        disabled=st.session_state['file_uploader_disabled_3'],
        on_change=handle_file_uploader_change_tab3,
        label_visibility="collapsed"
    )
    input_path_3 = col2.text_area(
        "   輸入資料夾路徑",
        height=68,
        key='text_area_' + str(st.session_state['text_area_key3']),
        disabled=st.session_state['text_area_disabled_3'],
        on_change=handle_text_area_change_tab3,
        placeholder="  輸入分包資料夾路徑",
        label_visibility="collapsed"
    )

    if uploaded_file_3 or input_path_3:
        tmpdirname = st.session_state["custom_tmpdir"]
        if not st.session_state.get("source_loaded", False):
            if uploaded_file_3:
                with zipfile.ZipFile(uploaded_file_3) as zip_ref:
                    zip_ref.extractall(tmpdirname)
            elif input_path_3:
                if input_path_3.startswith("search-ms:"):
                    match = re.search(r'location:([^&]+)', input_path_3)
                    if match:
                        input_path_3 = re.sub(r'%3A', ':', match.group(1))
                        input_path_3 = re.sub(r'%5C', '\\\\', input_path_3)
                    else:
                        st.warning("無法解析 search-ms 路徑，請確認輸入格式。")
                if not os.path.exists(input_path_3):
                    st.error("指定的本地路徑不存在，請重新輸入。")
                    st.stop()
                else:
                    copytree_multithreaded(input_path_3, tmpdirname)
            st.session_state["source_loaded"] = True

        # 讀取 "編圖複檢結果" Excel
        excel_file_path = None
        for f in os.listdir(tmpdirname):
            if f.lower().endswith('.xlsx') and '編圖結果' in f:
                excel_file_path = os.path.join(tmpdirname, f)
                break
        if excel_file_path and os.path.exists(excel_file_path):
            excel_sheets = pd.read_excel(excel_file_path, sheet_name=None)
            if "編圖複檢結果" in excel_sheets:
                df_check = excel_sheets["編圖複檢結果"]
            else:
                st.error("Excel 中未找到 '編圖複檢結果' 工作表。")
                st.stop()
        else:
            st.error("未找到任何 Excel 檔案。")
            st.stop()

        # 從 "編圖複檢結果" 中取品牌
        if "品牌" in df_check.columns:
            brand = df_check["品牌"].dropna().iloc[0]
        else:
            brand = ""

        # 讀取對應品牌資料集
        dataset_root = r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\資料集"
        brand_dataset_path = os.path.join(dataset_root, brand)
        if not os.path.exists(brand_dataset_path):
            st.error(f"資料集路徑中找不到品牌資料夾：{brand_dataset_path}")
            st.stop()
        dataset_xlsx = None
        for file in os.listdir(brand_dataset_path):
            if file.lower().endswith(".xlsx"):
                dataset_xlsx = os.path.join(brand_dataset_path, file)
                break
        if not dataset_xlsx:
            st.error(f"在品牌資料夾中未找到任何 .xlsx 檔案：{brand_dataset_path}")
            st.stop()
        ds = pd.read_excel(dataset_xlsx, sheet_name=None)
        target_sheet = None
        for sheet_name, df in ds.items():
            if df.shape[1] >= 3:
                headers = list(df.columns[:3])
                if headers == ["商品分類", "角度", "編號"]:
                    target_sheet = df
                    break
        if target_sheet is None:
            st.error("在品牌資料集的 .xlsx 檔案中，未找到標題為 [商品分類, 角度, 編號] 的分頁。")
            st.stop()

        # 取得可用商品分類
        category_list = target_sheet["商品分類"].dropna().unique().tolist()
        category_list = [str(cat).strip() for cat in category_list if str(cat).strip() != ""]

        # 建立 (商品分類, 角度) -> 編號 的 map
        angle_map = {}
        for idx, row in target_sheet.iterrows():
            c_ = str(row["商品分類"]).strip()
            a_ = str(row["角度"]).strip()
            n_ = str(row["編號"]).strip()
            angle_map[(c_, a_)] = n_

        # =============================================================================
        # 載入圖片
        # =============================================================================
        top_level_folders = [
            name for name in os.listdir(tmpdirname)
            if os.path.isdir(os.path.join(tmpdirname, name))
            and not name.startswith(('_', '.'))
            and name != "tmp_others"
        ]
        for folder in top_level_folders:
            st.session_state['image_cache'].setdefault(folder, {})
            img_folder_path = os.path.join(tmpdirname, folder, '2-IMG')
            if not os.path.exists(img_folder_path):
                img_folder_path = os.path.join(tmpdirname, folder, '1-Main', 'All')
            tasks = []
            with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                if os.path.exists(img_folder_path):
                    image_files = get_outer_folder_images(img_folder_path)
                    for image_file in image_files:
                        image_path = os.path.join(img_folder_path, image_file)
                        if image_path not in st.session_state['image_cache'][folder]:
                            add_label = image_file.lower().endswith(('.png', '.tif', '.tiff', '.psd', '.ai'))
                            future = executor.submit(load_and_process_image, image_path, add_label)
                            tasks.append((future, folder, image_path))
                outer_folder_path = os.path.join(tmpdirname, folder)
                outer_images = get_outer_folder_images(outer_folder_path)
                for outer_image_file in outer_images:
                    image_path = os.path.join(outer_folder_path, outer_image_file)
                    if image_path not in st.session_state['image_cache'][folder]:
                        add_label = outer_image_file.lower().endswith(('.png', '.tif', '.tiff', '.psd', '.ai'))
                        future = executor.submit(load_and_process_image, image_path, add_label)
                        tasks.append((future, folder, image_path))
                for future, folder_, image_path_ in tasks:
                    try:
                        image_ = future.result()
                        st.session_state['image_cache'][folder_][image_path_] = image_
                    except Exception as e:
                        st.warning(f"載入圖片 {image_path_} 時發生錯誤: {str(e)}")

        # =============================================================================
        # 顯示資料夾與角度選擇介面
        # =============================================================================
        if top_level_folders:
            if 'previous_selected_folder' not in st.session_state and top_level_folders:
                st.session_state['previous_selected_folder'] = top_level_folders[0]
            if top_level_folders:
                if 'previous_selected_folder' not in st.session_state:
                    st.session_state['previous_selected_folder'] = None
                col1_, col2_ = st.columns([7, 1], vertical_alignment="bottom")
                selected_folder = col1_.pills(
                    "選擇一個資料夾",
                    top_level_folders,
                    default=top_level_folders[0],
                    label_visibility="collapsed",
                )

                st.session_state['previous_selected_folder'] = selected_folder
                if selected_folder is None:
                    st.stop()

                folder_records = df_check[df_check["圖片"].apply(lambda x: str(x).split("\\")[0] == selected_folder)]
                if not folder_records.empty:
                    default_category = folder_records.iloc[0]["商品分類"]
                else:
                    default_category = ""
                if "category_selection" not in st.session_state:
                    st.session_state["category_selection"] = {}
                if selected_folder in st.session_state["category_selection"]:
                    current_category = st.session_state["category_selection"][selected_folder]
                else:
                    current_category = default_category

                def update_category_callback(folder=selected_folder):
                    st.session_state["category_selection"][folder] = st.session_state[f"{folder}_category"]

                category = col2_.selectbox(
                    "商品分類",
                    options=category_list,
                    index=category_list.index(current_category) if current_category in category_list else 0,
                    key=f"{selected_folder}_category",
                    on_change=update_category_callback,
                    label_visibility="collapsed",
                )

                df_category = target_sheet[target_sheet["商品分類"] == category]
                angle_options_all = df_category["角度"].dropna().unique().tolist()
                angle_options_all = [str(a).strip() for a in angle_options_all if str(a).strip() != ""]
                angle_options_all = [a for a in angle_options_all if re.search(r'[\u4e00-\u9fff]', a)]
                saved_angles = st.session_state.get("angle_changes", {}).get(selected_folder, {})
                angle_keys = []

                with st.form(f"angle_form_{selected_folder}"):
                    cols = st.columns(7)
                    for idx, (_, row) in enumerate(folder_records.iterrows()):
                        if idx % 7 == 0 and idx != 0:
                            cols = st.columns(7)
                        col_ = cols[idx % 7]
                        image_relative_path = row["圖片"]
                        image_path = os.path.join(tmpdirname, image_relative_path)
                        if image_path in st.session_state['image_cache'][selected_folder]:
                            image_ = st.session_state['image_cache'][selected_folder][image_path]
                            col_.image(image_, use_container_width=True)
                        default_angle = saved_angles.get(image_relative_path, row["角度"] if pd.notna(row["角度"]) else "")
                        angle_options_with_skip = ["不訓練"] + angle_options_all
                        if default_angle == "不訓練":
                            default_index = 0
                        elif default_angle in angle_options_all:
                            default_index = angle_options_all.index(default_angle) + 1
                        else:
                            default_index = None
                        angle_key = f"{selected_folder}_angle_{image_relative_path}"
                        angle_keys.append((angle_key, image_relative_path))
                        col_.selectbox(
                            "角度",
                            options=angle_options_with_skip,
                            index=default_index,
                            key=angle_key,
                            label_visibility="collapsed",
                            placeholder=""
                        )
                    st.divider()
                    colA, colB, colC, colD = st.columns([3, 5, 7.5, 3], vertical_alignment="center")
                    colA.form_submit_button(
                        "暫存修改",
                        on_click=handle_submit_angles,
                        args=(selected_folder, angle_keys)
                    )
                    with colD.popover("訓練資料預覽"):
                        final_data = []
                        for idx_, row_ in df_check.iterrows():
                            url_ = row_["圖片"]
                            brand_ = row_["品牌"] if "品牌" in df_check.columns else ""
                            folder_ = str(url_).split("\\")[0]
                            if folder_ in st.session_state["category_selection"]:
                                final_category = st.session_state["category_selection"][folder_]
                            else:
                                final_category = row_.get("商品分類", "")
                            angle_key_ = f"{folder_}_angle_{url_}"
                            if angle_key_ in st.session_state:
                                final_angle = st.session_state[angle_key_]
                            elif folder_ in st.session_state.get("angle_changes", {}):
                                chosen_angles = st.session_state["angle_changes"][folder_]
                                if url_ in chosen_angles:
                                    final_angle = chosen_angles[url_]
                                else:
                                    final_angle = row_.get("角度", "")
                            else:
                                final_angle = row_.get("角度", "")
                            if pd.isna(final_angle) or str(final_angle).strip() == "" or final_angle == "不訓練":
                                continue
                            df_category_final = target_sheet[target_sheet["商品分類"] == str(final_category).strip()]
                            angle_options_final = df_category_final["角度"].dropna().unique().tolist()
                            angle_options_final = [str(a).strip() for a in angle_options_final if str(a).strip() != ""]
                            angle_options_final = [a for a in angle_options_final if re.search(r'[\u4e00-\u9fff]', a)]
                            if final_angle not in angle_options_final:
                                continue
                            final_number = angle_map.get((str(final_category).strip(), str(final_angle).strip()), "")
                            final_data.append({
                                "URL": url_,
                                "品牌": brand_,
                                "商品分類": final_category,
                                "角度": final_angle,
                                "編號": final_number
                            })
                        if final_data:
                            df_result = pd.DataFrame(final_data, columns=["URL", "品牌", "商品分類", "角度", "編號"])
                            st.dataframe(df_result, use_container_width=True, hide_index=True)
                            # 將訓練用的 df_result 存到 session_state 供後續使用
                            st.session_state["df_result"] = df_result
                        else:
                            st.warning("沒有任何符合條件的圖片資料。")
        else:
            st.error("未找到任何資料夾。")

        # =============================================================================
        # ResNet 訓練 (特徵萃取) 邏輯
        # =============================================================================
        if st.checkbox("所有資料夾均確認完成"):
            button_placeholder = st.empty()
            with button_placeholder:
                start_running = st.button('開始執行')
            if start_running:
                button_placeholder.empty()
                st.write("\n")
                # 改為直接使用上方 popover 產生的 df_result
                if "df_result" not in st.session_state or st.session_state["df_result"].empty:
                    st.warning("沒有任何可訓練的資料.")
                    st.stop()

                df_final = st.session_state["df_result"]

                # ---------------------
                # 2) 初始化模型與預處理
                # ---------------------
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
                
                weights = ResNet50_Weights.DEFAULT
                resnet = models.resnet50(weights=weights)
                resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
                resnet.eval().to(device)

                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])

                progress_bar = st.progress(0)
                progress_text = st.empty()
                total = len(df_final)
                processed_count = 0

                # 3) 依「df_final」資料，分品牌寫入各自的 pkl
                for brand_name in df_final["品牌"].unique():
                    brand_df = df_final[df_final["品牌"] == brand_name]

                    brand_folder = os.path.join(
                        r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies",
                        brand_name
                    )
                    os.makedirs(brand_folder, exist_ok=True)

                    pkl_file_path = os.path.join(brand_folder, f"{brand_name}_image_features.pkl")
                    features_by_category = load_existing_features(pkl_file_path)

                    # 先計算舊有 labeled_features 數量
                    original_len = 0
                    if brand_name in features_by_category:
                        for cat_key, cat_data in features_by_category[brand_name].items():
                            original_len += len(cat_data["labeled_features"])

                    for idx, row_ in brand_df.iterrows():
                        image_path = os.path.join(tmpdirname, row_["URL"])  # 以 df_result 中的相對路徑來取得完整路徑
                        category_ = row_["商品分類"]
                        angle_ = row_["角度"]
                        number_ = row_["編號"]

                        if not os.path.exists(image_path):
                            st.warning(f"找不到檔案：{image_path}")
                            processed_count += 1
                            progress_bar.progress(int(processed_count / total * 100))
                            progress_text.write(f"正在處理 {processed_count} / {total}")
                            continue

                        try:
                            img = Image.open(image_path).convert('RGB')
                        except UnidentifiedImageError:
                            st.warning(f"無法識別圖片：{image_path}")
                            processed_count += 1
                            progress_bar.progress(int(processed_count / total * 100))
                            progress_text.write(f"正在處理 {processed_count} / {total}")
                            continue

                        image_features = get_image_features(img, resnet)

                        if brand_name not in features_by_category:
                            features_by_category[brand_name] = {}
                        if category_ not in features_by_category[brand_name]:
                            features_by_category[brand_name][category_] = {
                                "all_features": [],
                                "labeled_features": []
                            }

                        features_by_category[brand_name][category_]["all_features"].append(image_features)
                        features_by_category[brand_name][category_]["labeled_features"].append({
                            "features": image_features,
                            "labels": {
                                "brand": brand_name,
                                "category": category_,
                                "angle": angle_,
                                "number": number_
                            }
                        })

                        processed_count += 1
                        progress_bar.progress(int(processed_count / total * 100))
                        progress_text.write(f"正在處理 {processed_count} / {total}")

                    # 去重並寫回 pkl
                    progress_bar.empty()
                    progress_text.empty()

                    with st.spinner(" 尚未結束，請勿關閉"):
                        features_by_category, dupe_count = remove_duplicates(features_by_category)
                        with open(pkl_file_path, 'wb') as f:
                            pickle.dump(features_by_category, f)

                        # 重新計算去重後的 labeled_features 總數量
                        new_len = 0
                        if brand_name in features_by_category:
                            for cat_key, cat_data in features_by_category[brand_name].items():
                                new_len += len(cat_data["labeled_features"])

                        added_count = new_len - original_len

                        if dupe_count > 0:
                            st.success(f"{brand_name}圖片資料訓練完成！共新增{added_count}筆資料 (發現{dupe_count}筆重複資料)")
                        else:
                            st.success(f"{brand_name}圖片資料訓練完成！共新增{added_count}筆資料")

                if st.button("結束", on_click=reset_tab3()):
                    st.rerun()