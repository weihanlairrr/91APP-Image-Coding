import streamlit as st
import pandas as pd
import zipfile
import os
import shutil
import re
import tempfile
import functools
import imagecodecs
import ctypes
import time
import subprocess
import sys
import stat
import concurrent.futures
import pymupdf
from io import BytesIO
from psd_tools import PSDImage
from collections import Counter, defaultdict
from PIL import Image, ImageOps, ImageDraw, ImageFont, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed

def tab2():
    # =============================================================================
    # 常數集中
    # =============================================================================
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".psd", ".ai")
    LABEL_EXTENSIONS = (".png", ".tif", ".tiff", ".psd", ".ai")
    TMP_OTHERS_DIRNAME = "tmp_others"
    MAIN_DIR = "1-Main"
    IMG_DIR = "2-IMG"
    EXCEL_KEYWORD = "編圖結果"
    SHEET_COUNT = "編圖張數與廣告圖"
    SHEET_TYPES = "圖片類型統計"
    SHEET_RECORD = "編圖紀錄"

    # =============================================================================
    # 協助函式 - 檔案與目錄
    # =============================================================================
    def on_rm_error(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def copytree_multithreaded(src, dst):
        """快速大量複製：Windows 用 robocopy、Unix 用 rsync；失敗再改 ThreadPool 逐檔案複製。"""
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

    def setup_temporary_directory(base_path, tmp_dir):
        """
        建立「其他檔案暫存夾」：複製除圖片以外的結構與檔案，並移除每個 Top-level 子包內的 1-Main 或 2-IMG。
        """
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        for root, dirs, files in os.walk(base_path):
            if tmp_dir in root:
                continue
            is_top_level = os.path.basename(root) in [MAIN_DIR, IMG_DIR]
            is_same_level_as_image_folders = os.path.dirname(root) == base_path

            # 複製目錄結構（排除與 1-Main / 2-IMG 同層且其本身）
            for item in dirs:
                item_path = os.path.join(root, item)
                relative_path = os.path.relpath(item_path, base_path)
                if not (is_same_level_as_image_folders and is_top_level):
                    dest_path = os.path.join(tmp_dir, relative_path)
                    os.makedirs(dest_path, exist_ok=True)

            # 複製非圖片檔
            for item in files:
                item_path = os.path.join(root, item)
                relative_path = os.path.relpath(item_path, base_path)
                ext = os.path.splitext(item)[1].lower()
                if is_same_level_as_image_folders and ext in IMAGE_EXTENSIONS:
                    continue
                dest_path = os.path.join(tmp_dir, relative_path)
                try:
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(item_path, dest_path)
                except FileNotFoundError as e:
                    st.warning(f"無法建立路徑：{dest_path}，錯誤：{str(e)}")

        # 移除 top-level 子包內的 1-Main 或 2-IMG 夾，使 tmp_others 僅保留「其它檔案」
        for folder_name in os.listdir(tmp_dir):
            folder_path = os.path.join(tmp_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            two_img_path = os.path.join(folder_path, IMG_DIR)
            if os.path.exists(two_img_path):
                shutil.rmtree(two_img_path)
            elif os.path.exists(os.path.join(folder_path, MAIN_DIR)):
                shutil.rmtree(os.path.join(folder_path, MAIN_DIR))

    def get_outer_folder_images(folder_path):
        return sorted([f for f in os.listdir(folder_path)
                       if f.lower().endswith(IMAGE_EXTENSIONS)])

    def get_prefix(image_files):
        """從檔名中抓第一個底線前綴（含底線），例如 ABC_01.jpg -> 'ABC_'。"""
        for image_file in image_files:
            filename_without_ext = os.path.splitext(image_file)[0]
            first_underscore_index = filename_without_ext.find('_')
            if first_underscore_index != -1:
                return filename_without_ext[:first_underscore_index + 1]
        return ""

    def write_to_zip(zipf, file_path, arcname):
        """根據檔案型態選擇壓縮方式寫入 ZIP。"""
        if os.path.splitext(arcname)[1].lower() == '.db':
            return
        ct = get_compress_type(file_path)
        if ct == zipfile.ZIP_DEFLATED:
            zipf.write(file_path, arcname, compress_type=ct, compresslevel=1)
        else:
            zipf.write(file_path, arcname, compress_type=ct)

    def get_compress_type(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return zipfile.ZIP_STORED
        else:
            return zipfile.ZIP_DEFLATED

    # =============================================================================
    # 協助函式 - Excel 與對應表
    # =============================================================================
    def zfill2_int_str(x):
        """將數字字串補成兩位（例：'1'->'01'），非數字則原樣回傳。"""
        try:
            return f"{int(x):02}"
        except Exception:
            return str(x)

    def fix_code(val):
        """乾淨化欄位：NaN->空字串，去前後空白。"""
        if pd.isna(val):
            return ""
        return str(val).strip()

    def fix_code2(val):
        """同 fix_code，保留命名一致以不影響原流程。"""
        if pd.isna(val):
            return ""
        return str(val).strip()

    def find_excel_in_dir(base_dir):
        """尋找包含關鍵字『編圖結果』的 .xlsx 檔。"""
        excel_file_path, found_excel_name = None, None
        for f in os.listdir(base_dir):
            if f.lower().endswith('.xlsx') and EXCEL_KEYWORD in f:
                excel_file_path = os.path.join(base_dir, f)
                found_excel_name = f
                break
        return excel_file_path, found_excel_name

    def read_sheets_dict(excel_file_path):
        """讀取全部工作表；不存在則回傳空結構。"""
        if excel_file_path and os.path.exists(excel_file_path):
            return pd.read_excel(excel_file_path, sheet_name=None)
        return {}

    def load_df_result_and_mapping_from_tmpdir():
        """
        從暫存資料夾找『編圖結果.xlsx』，讀取『編圖紀錄』並建立 (folder, prefix+number 或 number) -> 原始檔名(無附檔名) 的對應表。
        兩個 submit_* 皆共用，避免重複程式碼。
        """
        df_result, mapping = None, {}
        excel_file_path, _ = find_excel_in_dir(st.session_state["custom_tmpdir"])
        if excel_file_path and os.path.exists(excel_file_path):
            try:
                # 指定 dtype，確保「前綴」與「編號」為字串
                sheets = pd.read_excel(excel_file_path, sheet_name=None, dtype={'前綴': str, '編號': str})
                if SHEET_RECORD in sheets:
                    df_result = sheets[SHEET_RECORD]
                    tmp_map = {}
                    for idx, row in df_result.iterrows():
                        folder_name = str(row.iloc[0]).strip()
                        original_full_filename = str(row.iloc[2]).strip()
                        orig_filename_only = os.path.basename(original_full_filename.replace("\\", "/"))
                        orig_text = os.path.splitext(orig_filename_only)[0]
                        prefix_val = fix_code(row.iloc[5])
                        number_val = fix_code(row.iloc[6])
                        key = (folder_name, number_val if prefix_val == "" else prefix_val + number_val)
                        tmp_map[key] = orig_text
                    mapping = tmp_map
                else:
                    df_result = None
            except Exception:
                df_result = None
        return df_result, mapping

    def get_original_filename(selected_folder, image_file, df_result, mapping):
        """依『編圖紀錄』建立的 mapping 回推原始檔名（無附檔名）；若無對應回傳現名(無附檔名)。"""
        norm_folder = selected_folder[:-3] if selected_folder.endswith("_OK") else selected_folder
        current_filename = os.path.splitext(os.path.basename(image_file))[0]
        if df_result is not None:
            key = (norm_folder, current_filename)
            return mapping.get(key, current_filename)
        else:
            return current_filename

    def get_image_label(folder, image_file, df):
        """
        依『編圖紀錄』中描述判斷圖片標籤：模特 / 平拍 / 無。
        注意：兩段判斷的關鍵字集合刻意與原版一致（第二段沒有「上腳」），以維持行為完全相同。
        """
        norm_folder = folder[:-3] if folder.endswith("_OK") else folder
        df_folder = df[df.iloc[:, 0].astype(str).str.strip() == norm_folder]
        base_name = os.path.splitext(image_file)[0]

        # 第一輪：比對『前綴+編號』
        for _, row in df_folder.iterrows():
            prefix_val = fix_code(row.iloc[5])
            number_val = fix_code(row.iloc[6])
            combined = prefix_val + number_val
            if combined == base_name:
                description = str(row.iloc[4])
                return "模特" if any(keyword in description for keyword in ["模特", "_9", "-0m", "上腳"]) else "平拍"

        # 第二輪：比對『原始檔名(不含副檔名)』是否包含 base_name
        for _, row in df_folder.iterrows():
            original_full_filename = str(row.iloc[2]).strip()
            orig_filename_only = os.path.basename(original_full_filename.replace("\\", "/"))
            orig_text = os.path.splitext(orig_filename_only)[0]
            if base_name in orig_text:
                description = str(row.iloc[4])
                return "模特" if any(keyword in description for keyword in ["模特", "_9", "-0m"]) else "平拍"

        return "無"

    def ensure_original_and_label_state(selected_folder, images, df_result, mapping):
        """
        將 original_filename 與 image_labels 兩個 session_state 統一初始化（若尚未建立）。
        由 submit_main_folder / submit_img_folder 共用。
        """
        st.session_state.setdefault('original_filename', {})
        st.session_state['original_filename'].setdefault(selected_folder, {})
        st.session_state.setdefault('image_labels', {})
        st.session_state['image_labels'].setdefault(selected_folder, {})

        for image_file in images:
            if image_file not in st.session_state['original_filename'][selected_folder]:
                st.session_state['original_filename'][selected_folder][image_file] = get_original_filename(
                    selected_folder, image_file, df_result, mapping
                )
            if image_file not in st.session_state['image_labels'][selected_folder]:
                st.session_state['image_labels'][selected_folder][image_file] = (
                    get_image_label(selected_folder, image_file, df_result) if df_result is not None else "無"
                )

    # =============================================================================
    # 協助函式 - 影像載入與視覺標示
    # =============================================================================
    def add_image_label(image, file_extension):
        draw = ImageDraw.Draw(image)
        label_map = {'.png': 'PNG', '.tif': 'TIF', '.tiff': 'TIF', '.psd': 'PSD', '.ai': 'AI'}
        label_text = label_map.get(file_extension.lower())
        if not label_text:
            return image
        font_size = max(30, int(image.width * 0.12))
        try:
            if sys.platform == 'darwin':
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
            else:
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
        """通用載圖 + 1000x1000 pad + 透明處理 +（選用）左上角副檔名標籤。"""
        psd_cache = psd_cache_dir
        ai_cache = ai_cache_dir
        ext = os.path.splitext(image_path)[1].lower()
    
        # ---------------- PSD ----------------
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
                    st.warning(f"⚠️ 無法處理 PSD 檔案：{os.path.basename(image_path)}，已跳過。")
                    return None
    
        # ---------------- AI ----------------
        elif ext == '.ai':
            os.makedirs(ai_cache, exist_ok=True)
            cache_file_name = str(abs(hash(image_path))) + ".png"
            cache_path = os.path.join(ai_cache, cache_file_name)
            if os.path.exists(cache_path):
                image = Image.open(cache_path)
            else:
                try:
                    doc = pymupdf.open(image_path)
                    page = doc.load_page(0)
                    pix = page.get_pixmap(dpi=100)
                    image = Image.open(BytesIO(pix.tobytes("png")))
                    doc.close()
                    image.save(cache_path, format='PNG')
                except Exception as e:
                    st.warning(f"⚠️ 無法處理 AI 檔案：{os.path.basename(image_path)}，已跳過（錯誤：{e}）")
                    return None
    
        # ---------------- 其他格式（JPG / PNG / BMP / TIFF ...） ----------------
        else:
            try:
                image = Image.open(image_path)
            except UnidentifiedImageError:
                # 僅在 .tif / .tiff 時嘗試 tiff_decode，其餘視為損壞
                if ext in ('.tif', '.tiff'):
                    try:
                        with open(image_path, 'rb') as f:
                            raw_data = f.read()
                        decoded_data = imagecodecs.tiff_decode(raw_data)
                        image = Image.fromarray(decoded_data)
                    except Exception as e:
                        st.warning(f"⚠️ 無法讀取 TIFF 檔案：{os.path.basename(image_path)}，已跳過（錯誤：{e}）")
                        return None
                else:
                    st.warning(f"⚠️ 圖片毀損或格式不支援：{os.path.basename(image_path)}，已跳過。")
                    return None
            except Exception as e:
                st.warning(f"⚠️ 載入圖片時發生錯誤：{os.path.basename(image_path)}，已跳過（錯誤：{e}）")
                return None
    
        # ---------------- 透明 / PAD / 標籤 ----------------
        try:
            if image.mode in ('RGBA', 'LA') or (hasattr(image, 'info') and image.info.get('transparency')):
                image = image.convert('RGBA')
                pad_color = (255, 255, 255, 0)
            else:
                image = image.convert('RGB')
                pad_color = (255, 255, 255)
            image = ImageOps.pad(image, (1000, 1000), method=Image.Resampling.LANCZOS, color=pad_color)
    
            if add_label:
                image = add_image_label(image, ext)
            return image
    
        except Exception as e:
            st.warning(f"⚠️ 處理圖片時發生錯誤：{os.path.basename(image_path)}，已跳過（錯誤：{e}）")
            return None

    def get_sort_key(image_file):
        """
        排序 key：維持原先行為（先長度、是否字母開頭、首字母、第一個數字、完整字串）。
        注意：存取外層變數 selected_folder（Python 以 closure 捕捉，行為與原碼一致）。
        """
        filename_changes = st.session_state.get('filename_changes', {}).get(selected_folder, {})
        if image_file in filename_changes:
            new_filename = filename_changes[image_file]['new_filename']
            filename = new_filename if new_filename else image_file
        else:
            filename = image_file
        base_name = os.path.splitext(filename)[0]
        length_priority = 0 if len(base_name) <= 5 else 1
        is_alpha = 0 if (base_name and base_name[0].isalpha()) else 1
        first_letter = base_name[0].upper() if base_name and base_name[0].isalpha() else ''
        match = re.search(r'(\d+)', base_name)
        num = int(match.group(1)) if match else float('inf')
        return (length_priority, is_alpha, first_letter, num, base_name)

    # =============================================================================
    # 協助函式 - UI 觸發（上傳 / 文字框）
    # =============================================================================
    def initialize_tab2(mode="default"):
        """
        初始化與重置 session_state 和暫存資料夾。
        mode:
          - "default": 僅設定預設值（若 session_state 中無該鍵則設定）
          - "clear_cache": 清除快取資料夾
          - "reinitialize": 清除指定鍵與所有 "prev_" 開頭的鍵
          - "both": 同時清快取+重置狀態，再做預設初始化
        """
        if mode in ("clear_cache", "both"):
            if os.path.exists(cache_base_dir):
                shutil.rmtree(cache_base_dir, ignore_errors=False, onerror=on_rm_error)
            os.makedirs(cache_base_dir, exist_ok=True)
            os.makedirs(psd_cache_dir, exist_ok=True)
            os.makedirs(ai_cache_dir, exist_ok=True)
            os.makedirs(custom_tmp_dir, exist_ok=True)

        if mode in ("reinitialize", "both"):
            keys = [
                'filename_changes', 'image_cache', 'folder_values', 'confirmed_changes',
                'uploaded_file_name', 'last_text_inputs', 'has_duplicates', 'duplicate_filenames',
                'modified_folders', 'previous_uploaded_file_name', 'previous_input_path',
                'file_uploader_disabled_2', 'text_area_disabled_2', 'custom_tmpdir',
                'previous_selected_folder', 'final_zip_content', 'source_loaded',
                'original_filename', 'image_labels', 'outer_images_map', 'processing_complete'
            ]
            for key in keys:
                if key in st.session_state:
                    del st.session_state[key]
            for key in list(st.session_state.keys()):
                if key.startswith("prev_"):
                    del st.session_state[key]

        defaults = {
            'filename_changes': {},
            'image_cache': {},
            'folder_values': {},
            'confirmed_changes': {},
            'uploaded_file_name': None,
            'last_text_inputs': {},
            'has_duplicates': False,
            'duplicate_filenames': [],
            'file_uploader_key2': 20,
            'text_area_key2': 20,
            'modified_folders': set(),
            'previous_uploaded_file_name': None,
            'previous_input_path': None,
            'file_uploader_disabled_2': False,
            'text_area_disabled_2': False,
            'custom_tmpdir': custom_tmp_dir,
            'previous_selected_folder': None,
            'final_zip_content': None,
            'source_loaded': False,
            'original_filename': {},
            'image_labels': {},
            'outer_images_map': {},
            'processing_complete': False,
        }
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)

    def handle_file_uploader_change_tab2():
        initialize_tab2(mode="both")
        st.session_state["custom_tmpdir"] = custom_tmp_dir
        file_key = 'file_uploader_' + str(st.session_state.get('file_uploader_key2', 0))
        uploaded_file_1 = st.session_state.get(file_key, None)
        if uploaded_file_1:
            st.session_state['image_cache'].clear()
            st.session_state['filename_changes'].clear()
            st.session_state['confirmed_changes'].clear()
            st.session_state['folder_values'].clear()
            st.session_state['previous_uploaded_file_name'] = uploaded_file_1.name
            st.session_state["source_loaded"] = False
        st.session_state.text_area_disabled_2 = bool(uploaded_file_1)

    def handle_text_area_change_tab2():
        initialize_tab2(mode="both")
        st.session_state["custom_tmpdir"] = custom_tmp_dir
        text_key = 'text_area_' + str(st.session_state.get('text_area_key2', 0))
        text_content = st.session_state.get(text_key, "").strip()
        st.session_state['image_cache'].clear()
        st.session_state['filename_changes'].clear()
        st.session_state['confirmed_changes'].clear()
        st.session_state['folder_values'].clear()
        st.session_state['previous_input_path'] = text_content
        st.session_state["source_loaded"] = False
        st.session_state.file_uploader_disabled_2 = bool(text_content)

    # =============================================================================
    # 協助函式 - 提交計算（主流程相依）
    # =============================================================================
    def submit_main_folder(selected_folder, desired_images, excluded_images, folder_stats_map):
        """
        適用 1-Main/All 結構：處理內層 desired 與外層 excluded 的命名與計數；無重複時自動序號化 01,02...
        """
        def sort_key(item):
            text = item[1]['text'].strip()
            if text.isdigit():
                return (0, int(text))
            return (1, text)

        added_image_count = removed_image_count = 0
        added_model_count = removed_model_count = 0
        added_flat_count = removed_flat_count = 0

        df_result, mapping = load_df_result_and_mapping_from_tmpdir()
        prefix = get_prefix(desired_images)

        # 準備原始檔名/標籤快取（desired + excluded）
        ensure_original_and_label_state(selected_folder, desired_images + excluded_images, df_result, mapping)

        current_filenames = {}
        temp_filename_changes = {}

        # 內層圖片（desired_images）
        for image_file in desired_images:
            text_input_key = f"{selected_folder}_{image_file}"
            new_text = st.session_state.get(text_input_key, "")
            extension = os.path.splitext(image_file)[1]
            prev_key = f"prev_{text_input_key}"
            prev_text = st.session_state.get(prev_key, None)
            original_filename = st.session_state['original_filename'][selected_folder].get(image_file, "無")
            image_label = st.session_state['image_labels'][selected_folder].get(image_file, "無")

            if prev_text is None:
                filename_without_ext = os.path.splitext(image_file)[0]
                first_underscore_index = filename_without_ext.find('_')
                prev_text = filename_without_ext[first_underscore_index + 1:] if first_underscore_index != -1 else filename_without_ext

            if new_text.strip() == '':
                if prev_text != '':
                    removed_image_count += 1
                    if image_label == "模特":
                        removed_model_count += 1
                    elif image_label == "平拍":
                        removed_flat_count += 1
                new_text = original_filename
                current_filename = original_filename
                new_filename = ""
            else:
                new_filename = prefix + new_text + extension
                current_filename = new_text

            current_filenames[image_file] = {'new_filename': new_filename, 'text': current_filename}
            temp_filename_changes[image_file] = {'new_filename': new_filename, 'text': current_filename}
            st.session_state[prev_key] = current_filename

        # 外層圖片（excluded_images）
        for outer_image_file in excluded_images:
            text_input_key = f"outer_{selected_folder}_{outer_image_file}"
            new_text = st.session_state.get(text_input_key, "")
            extension = os.path.splitext(outer_image_file)[1]
            prev_key = f"prev_{text_input_key}"
            prev_text = st.session_state.get(prev_key, None)
            original_filename = st.session_state['original_filename'][selected_folder].get(outer_image_file, "無")
            image_label = st.session_state['image_labels'][selected_folder].get(outer_image_file, "無")

            if f"outer_{selected_folder}_{outer_image_file}" in st.session_state:
                del st.session_state[f"outer_{selected_folder}_{outer_image_file}"]

            if prev_text is None:
                filename_without_ext = os.path.splitext(outer_image_file)[0]
                first_underscore_index = filename_without_ext.find('_')
                prev_text = filename_without_ext[first_underscore_index + 1:] if first_underscore_index != -1 else filename_without_ext

            current_filename = new_text

            if new_text.strip() == '':
                current_filename = prev_text
                new_text = current_filename
                new_filename = current_filename + extension
            elif new_text.strip() == original_filename:
                if prev_text == original_filename or current_filename == original_filename:
                    new_text = original_filename
                    prev_text = original_filename
                    current_filename = original_filename
                    new_filename = current_filename + extension
                else:
                    if prev_text != '':
                        removed_image_count += 1
                        if image_label == "模特":
                            removed_model_count += 1
                        elif image_label == "平拍":
                            removed_flat_count += 1
                    current_filename = prev_text
                    new_filename = prefix + current_filename + extension
            else:
                current_filename = new_text
                new_filename = prefix + current_filename + extension

            if new_text.strip() != prev_text:
                temp_filename_changes[outer_image_file] = {'new_filename': new_filename, 'text': current_filename}
                if new_filename != '':
                    added_image_count += 1
                    if image_label == "模特":
                        added_model_count += 1
                    elif image_label == "平拍":
                        added_flat_count += 1
            st.session_state[prev_key] = current_filename

        # 重複檢查 / 自動序號化
        new_filenames = [d['new_filename'] for d in temp_filename_changes.values() if d['new_filename'] != '']
        duplicates = [fn for fn, cnt in Counter(new_filenames).items() if cnt > 1]
        if duplicates:
            st.session_state['has_duplicates'] = True
            st.session_state['duplicate_filenames'] = duplicates
            st.session_state['confirmed_changes'][selected_folder] = False
        else:
            st.session_state['has_duplicates'] = False
            st.session_state['confirmed_changes'][selected_folder] = True
            sorted_files = sorted(((file, data) for file, data in temp_filename_changes.items() if data['new_filename'] != ''), key=sort_key)
            rename_counter = 1
            for file, data in sorted_files:
                new_index = str(rename_counter).zfill(2)
                extension = os.path.splitext(file)[1]
                new_filename = f"{get_prefix(desired_images)}{new_index}{extension}"
                temp_filename_changes[file]['new_filename'] = new_filename
                temp_filename_changes[file]['text'] = new_index
                rename_counter += 1

        # 狀態更新
        st.session_state.setdefault('filename_changes', {})
        if selected_folder not in st.session_state['filename_changes']:
            st.session_state['filename_changes'][selected_folder] = {}
        st.session_state['filename_changes'][selected_folder].update(temp_filename_changes)
        for file, data in temp_filename_changes.items():
            text_input_key = f"{selected_folder}_{file}"
            st.session_state[text_input_key] = data['text']

        # 影像張數/類型統計更新
        num_images_state_key = f"{selected_folder}_num_images"
        model_count_state_key = f"{selected_folder}_model_images"
        flat_count_state_key = f"{selected_folder}_flat_images"

        if num_images_state_key in st.session_state:
            current_num_images = int(st.session_state[num_images_state_key])
            st.session_state[num_images_state_key] = str(max(0, current_num_images - removed_image_count + added_image_count))

        current_model = int(st.session_state.get(model_count_state_key, 0))
        current_flat = int(st.session_state.get(flat_count_state_key, 0))
        st.session_state[model_count_state_key] = str(max(0, current_model - removed_model_count + added_model_count))
        st.session_state[flat_count_state_key] = str(max(0, current_flat - removed_flat_count + added_flat_count))

        # folder_values 統計
        ad_images_state_key = f"{selected_folder}_ad_images"
        ad_images_value = st.session_state.get(ad_images_state_key)
        data = folder_stats_map.get(selected_folder, {})
        data_folder_name = data.get('資料夾', selected_folder)
        st.session_state['folder_values'][data_folder_name] = {
            '張數': st.session_state.get(num_images_state_key),
            '廣告圖': ad_images_value,
            '模特': st.session_state.get(model_count_state_key),
            '平拍': st.session_state.get(flat_count_state_key),
        }
        st.session_state['modified_folders'].add(data_folder_name)

    def submit_img_folder(selected_folder, desired_images, excluded_images, folder_stats_map):
        """
        適用 2-IMG 結構：維持原行為（101~150 特區處理、非重複即保留文字命名，不做自動 01,02 序號化）。
        """
        added_image_count = removed_image_count = 0
        added_model_count = removed_model_count = 0
        added_flat_count = removed_flat_count = 0

        df_result, mapping = load_df_result_and_mapping_from_tmpdir()
        ensure_original_and_label_state(selected_folder, desired_images, df_result, mapping)

        current_filenames = {}
        temp_filename_changes = {}

        for image_file in desired_images:
            text_input_key = f"{selected_folder}_{image_file}"
            new_text = st.session_state.get(text_input_key, "")
            extension = os.path.splitext(image_file)[1]
            original_filename = st.session_state['original_filename'][selected_folder].get(image_file, "無")

            if (selected_folder in st.session_state.get('filename_changes', {}) and
                image_file in st.session_state['filename_changes'][selected_folder]):
                current_filename = st.session_state['filename_changes'][selected_folder][image_file]['text']
            else:
                current_filename = os.path.splitext(os.path.basename(image_file))[0]

            new_text_is_101 = new_text.strip().isdigit() and 101 <= int(new_text.strip()) <= 150
            current_filename_is_101 = current_filename.strip().isdigit() and 101 <= int(current_filename.strip()) <= 150

            if new_text.strip() == '':
                if original_filename != "無":
                    if current_filename != original_filename:
                        new_text = original_filename
                        new_filename = original_filename + extension
                        if not current_filename_is_101:
                            removed_image_count += 1
                            label_ = st.session_state['image_labels'][selected_folder].get(image_file, "無")
                            if label_ == "模特":
                                removed_model_count += 1
                            elif label_ == "平拍":
                                removed_flat_count += 1
                    else:
                        new_text = original_filename
                        new_filename = original_filename + extension
                else:
                    new_text = current_filename
                    new_filename = current_filename + extension
            else:
                if new_text.strip() != current_filename and not new_text_is_101:
                    if current_filename == original_filename or current_filename_is_101:
                        added_image_count += 1
                        label_ = st.session_state['image_labels'][selected_folder].get(image_file, "無")
                        if label_ == "模特":
                            added_model_count += 1
                        elif label_ == "平拍":
                            added_flat_count += 1
                elif new_text.strip() != current_filename and new_text_is_101:
                    if current_filename == original_filename or current_filename_is_101:
                        pass
                    else:
                        removed_image_count += 1
                        label_ = st.session_state['image_labels'][selected_folder].get(image_file, "無")
                        if label_ == "模特":
                            removed_model_count += 1
                        elif label_ == "平拍":
                            removed_flat_count += 1
                new_filename = new_text + extension

            current_filenames[image_file] = {'new_filename': new_filename, 'text': new_text}
            temp_filename_changes[image_file] = {'new_filename': new_filename, 'text': new_text}

        # 重複檢查（無自動序號化）
        new_filenames = [d['new_filename'] for d in temp_filename_changes.values() if d['new_filename'] != '']
        duplicates = [fn for fn, cnt in Counter(new_filenames).items() if cnt > 1]
        if duplicates:
            st.session_state['has_duplicates'] = True
            st.session_state['duplicate_filenames'] = duplicates
            st.session_state['confirmed_changes'][selected_folder] = False
        else:
            st.session_state['has_duplicates'] = False
            st.session_state['confirmed_changes'][selected_folder] = True

            num_images_state_key = f"{selected_folder}_num_images"
            ad_images_state_key = f"{selected_folder}_ad_images"
            model_count_state_key = f"{selected_folder}_model_images"
            flat_count_state_key = f"{selected_folder}_flat_images"

            if num_images_state_key in st.session_state:
                current_num_images = int(st.session_state[num_images_state_key])
                st.session_state[num_images_state_key] = str(max(0, current_num_images - removed_image_count + added_image_count))

            current_model = int(st.session_state.get(model_count_state_key, 0))
            current_flat = int(st.session_state.get(flat_count_state_key, 0))
            st.session_state[model_count_state_key] = str(max(0, current_model - removed_model_count + added_model_count))
            st.session_state[flat_count_state_key] = str(max(0, current_flat - removed_flat_count + added_flat_count))

            st.session_state.setdefault('filename_changes', {})
            if selected_folder not in st.session_state['filename_changes']:
                st.session_state['filename_changes'][selected_folder] = {}
            st.session_state['filename_changes'][selected_folder].update(temp_filename_changes)

            for file, data in temp_filename_changes.items():
                text_input_key = f"{selected_folder}_{file}"
                st.session_state[text_input_key] = data['text']

            ad_images_value = st.session_state.get(ad_images_state_key)
            data_ = folder_stats_map.get(selected_folder, {})
            data_folder_name = data_.get('資料夾', selected_folder)
            st.session_state['folder_values'][data_folder_name] = {
                '張數': st.session_state[num_images_state_key],
                '廣告圖': ad_images_value,
                '模特': st.session_state[model_count_state_key],
                '平拍': st.session_state[flat_count_state_key],
            }
            st.session_state['modified_folders'].add(data_folder_name)

    # =============================================================================
    # 協助函式 - ZIP 打包與清理
    # =============================================================================
    def merge_temporary_directory_to_zip(zipf, tmp_dir):
        """將 tmp_others 內容（含空目錄）寫入 ZIP。"""
        for root, dirs, files in os.walk(tmp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, tmp_dir)
                write_to_zip(zipf, file_path, relative_path)
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    relative_path = os.path.relpath(dir_path, tmp_dir)
                    zip_info = zipfile.ZipInfo(relative_path + "/")
                    zipf.writestr(zip_info, b"")

    def parallel_read_zip_compress():
        """
        建立 ZIP（未清理版）：
        - 檔案目錄重組（根據 filename_changes）
        - 讀檔並以 bytes 暫存後一次寫入 ZIP（壓縮策略依檔案型態）
        - 回寫 Excel（更新張數/廣告圖/類型統計/編號補零、加入『編圖複檢結果』）
        """
        file_entries = []

        # 1) 加入頂層非 Excel 檔
        top_level_files = [name for name in os.listdir(extract_dir) if os.path.isfile(os.path.join(extract_dir, name))]
        for file_name in top_level_files:
            file_path = os.path.join(extract_dir, file_name)
            arcname = file_name
            if file_name != f'{EXCEL_KEYWORD}.xlsx':
                file_entries.append((file_path, arcname))

        # 2) 依 filename_changes 重組各子包內檔案路徑
        for folder_name in top_level_folders:
            folder_path = os.path.join(extract_dir, folder_name)
            for root, dirs, files in os.walk(folder_path):
                if "_MACOSX" in root or TMP_OTHERS_DIRNAME in root:
                    continue
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, extract_dir)
                    path_parts = rel_path.split(os.sep)
                    original_file = file

                    if (folder_name in st.session_state['filename_changes']
                        and original_file in st.session_state['filename_changes'][folder_name]):
                        data = st.session_state['filename_changes'][folder_name][original_file]
                        new_filename = data['new_filename']

                        if new_filename.strip() == '':
                            # 被「去除」的 1-Main/All 圖：若外層 text 有值就用外層文字，否則回推原檔名
                            outer_key = f"outer_{folder_name}_{original_file}"
                            if (outer_key in st.session_state and st.session_state[outer_key].strip() != ""):
                                modified_filename = st.session_state[outer_key].strip() + os.path.splitext(original_file)[1]
                            else:
                                original_dict = st.session_state.get('original_filename', {}).get(folder_name, {})
                                real_original_name = original_dict.get(original_file, "").strip()
                                if real_original_name:
                                    modified_filename = real_original_name + os.path.splitext(original_file)[1]
                                else:
                                    modified_filename = original_file
                            new_rel_path = os.path.join(folder_name, modified_filename)
                        else:
                            # 被「保留/改名」的圖，放進 2-IMG 或 1-Main/All
                            if os.path.exists(os.path.join(extract_dir, folder_name, IMG_DIR)):
                                if len(path_parts) == 2:
                                    new_rel_path = os.path.join(folder_name, IMG_DIR, new_filename)
                                else:
                                    idx = path_parts.index(folder_name)
                                    path_parts = path_parts[:idx + 1] + [IMG_DIR, new_filename]
                                    new_rel_path = os.path.join(*path_parts)
                            else:
                                if len(path_parts) == 2:
                                    new_rel_path = os.path.join(folder_name, MAIN_DIR, "All", new_filename)
                                else:
                                    idx = path_parts.index(folder_name)
                                    path_parts = path_parts[:idx + 1] + [MAIN_DIR, "All", new_filename]
                                    new_rel_path = os.path.join(*path_parts)
                        file_entries.append((full_path, new_rel_path))
                    else:
                        file_entries.append((full_path, rel_path))

        # 3) 並行讀檔到記憶體
        file_bytes_map = {}
        with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
            futures = {executor.submit(lambda p: open(p, 'rb').read(), f[0]): f for f in file_entries}
            for future in as_completed(futures):
                fpath, arc = futures[future]
                try:
                    file_bytes_map[(fpath, arc)] = future.result()
                except Exception as e:
                    st.error(f"讀取檔案時發生錯誤 {fpath}: {str(e)}")

        # 4) 寫入 ZIP & 產生 Excel
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 4-1) 檔案寫入
            for (file_path, arcname), data_bytes in file_bytes_map.items():
                ct = get_compress_type(file_path)
                info = zipfile.ZipInfo(arcname)
                if ct == zipfile.ZIP_DEFLATED:
                    zipf.writestr(info, data_bytes, compress_type=ct, compresslevel=1)
                else:
                    zipf.writestr(info, data_bytes, compress_type=ct)

            # 4-2) 其他檔案合併（空目錄也保留）
            merge_temporary_directory_to_zip(zipf, tmp_dir_for_others)

            # 4-3) Excel 重寫（維持原行為）
            excel_bytes_buffer = BytesIO()
            if SHEET_RECORD in sheets_dict:
                df_result_local = sheets_dict[SHEET_RECORD]
                if "編號" in df_result_local.columns:
                    def _fix_code_num(x):
                        if pd.isna(x):
                            return ""
                        s = str(x)
                        if s.endswith(".0"):
                            s = s[:-2]
                        return s.zfill(2)
                    df_result_local["編號"] = df_result_local["編號"].apply(_fix_code_num)
                sheets_dict[SHEET_RECORD] = df_result_local

            if sheets_dict:
                # 更新『張數/廣告圖』
                result_df = sheets_dict.get(SHEET_COUNT, pd.DataFrame(columns=['資料夾', '張數', '廣告圖']))
                for idx, row in result_df.iterrows():
                    data_folder_name = str(row['資料夾'])
                    if data_folder_name in st.session_state['folder_values']:
                        num_images = st.session_state['folder_values'][data_folder_name]['張數']
                        ad_images = zfill2_int_str(st.session_state['folder_values'][data_folder_name]['廣告圖'])
                        result_df.at[idx, '張數'] = num_images
                        result_df.at[idx, '廣告圖'] = ad_images

                existing_folders = set(result_df['資料夾'])
                for data_folder_name in st.session_state['folder_values']:
                    if data_folder_name not in existing_folders:
                        num_images = st.session_state['folder_values'][data_folder_name]['張數']
                        ad_images = zfill2_int_str(st.session_state['folder_values'][data_folder_name]['廣告圖'])
                        new_row = pd.DataFrame([{'資料夾': data_folder_name, '張數': num_images, '廣告圖': ad_images}])
                        result_df = pd.concat([result_df, new_row], ignore_index=True)
                sheets_dict[SHEET_COUNT] = result_df

                # 更新『圖片類型統計』
                type_stats_df = sheets_dict.get(SHEET_TYPES, pd.DataFrame(columns=['資料夾', '模特', '平拍']))
                for idx, row in type_stats_df.iterrows():
                    data_folder_name = str(row['資料夾'])
                    if data_folder_name in st.session_state['folder_values']:
                        model_images = st.session_state['folder_values'][data_folder_name]['模特']
                        flat_images = st.session_state['folder_values'][data_folder_name]['平拍']
                        type_stats_df.at[idx, '模特'] = model_images
                        type_stats_df.at[idx, '平拍'] = flat_images

                existing_type_folders = set(type_stats_df['資料夾'])
                for data_folder_name in st.session_state['folder_values']:
                    if data_folder_name not in existing_type_folders:
                        model_images = st.session_state['folder_values'][data_folder_name]['模特']
                        flat_images = st.session_state['folder_values'][data_folder_name]['平拍']
                        new_row = pd.DataFrame([{'資料夾': data_folder_name, '模特': model_images, '平拍': flat_images}])
                        type_stats_df = pd.concat([type_stats_df, new_row], ignore_index=True)
                sheets_dict[SHEET_TYPES] = type_stats_df

                # 建立『編圖複檢結果』
                two_img_pattern = re.compile(r'^[A-Za-z][A-Za-z0-9]{0,4}$')

                def _record_map_from_sheets(sheets_dict_):
                    record_map_local = {}
                    if SHEET_RECORD in sheets_dict_:
                        df_record = sheets_dict_[SHEET_RECORD].fillna("")
                        for _, row in df_record.iterrows():
                            folder_name = fix_code2(row.iloc[0])
                            subfolder = fix_code2(row.iloc[1])
                            original_file_col = fix_code2(row.iloc[2])
                            category = fix_code2(row.iloc[3])
                            angle = fix_code2(row.iloc[4])
                            prefix_val = fix_code2(row.iloc[5])
                            number_val = fix_code2(row.iloc[6])
                            ext_ = os.path.splitext(original_file_col)[1]
                            original_filename = f"{prefix_val}{number_val}{ext_}" if number_val else original_file_col
                            norm_folder = folder_name.strip()
                            record_map_local[(norm_folder, original_filename)] = (folder_name, subfolder, category, angle)
                    return record_map_local

                record_map = _record_map_from_sheets(sheets_dict)

                brand_name = ""
                if found_excel_name and found_excel_name.endswith(f"{EXCEL_KEYWORD}.xlsx"):
                    brand_name = found_excel_name[:-len(f"{EXCEL_KEYWORD}.xlsx")]

                review_records = []
                for (_, arc) in file_entries:
                    p = arc.split(os.sep)
                    if len(p) < 2:
                        continue
                    fn_ = p[-1]
                    bn, ext = os.path.splitext(fn_)
                    is_main_all = False
                    is_2_img = False
                    if len(p) >= 3:
                        if p[1].lower() == '1-main' and p[2].lower() == 'all':
                            is_main_all = True
                        elif p[1].lower() == '2-img':
                            is_2_img = True
                    elif len(p) == 2:
                        if p[1].lower() == '2-img':
                            is_2_img = True
                        elif '1-main' in p[1].lower():
                            is_main_all = True
                    if is_main_all:
                        pass
                    elif is_2_img:
                        if not two_img_pattern.match(bn):
                            continue
                    else:
                        continue

                    folder_nm = p[0]
                    norm_folder = folder_nm[:-3] if folder_nm.endswith("_OK") else folder_nm

                    # 嘗試由 filename_changes 找回原始舊名
                    org_name = None
                    chg = st.session_state['filename_changes'].get(folder_nm, {})
                    for ofile, d in chg.items():
                        if d['new_filename'] == fn_:
                            org_name = ofile
                            break
                    if org_name is None:
                        org_name = fn_

                    key_ = (norm_folder, org_name)
                    if key_ not in record_map:
                        continue
                    fv, sv, catv, angv = record_map[key_]
                    if sv.replace('/', '\\').lower() == '1-main\\all':
                        fv_ok = fv + "_OK"
                    else:
                        fv_ok = fv
                    sv_ = sv.replace('/', '\\').strip()
                    if sv_:
                        full_path_str = f"{fv_ok}\\{sv_}\\{fn_}"
                    else:
                        full_path_str = f"{fv_ok}\\{fn_}"
                    review_records.append({
                        "圖片": full_path_str,
                        "品牌": brand_name,
                        "商品分類": catv,
                        "角度": angv
                    })

                review_df = pd.DataFrame(review_records, columns=["圖片", "品牌", "商品分類", "角度"])
                sheets_dict["編圖複檢結果"] = review_df

                with pd.ExcelWriter(excel_bytes_buffer, engine='xlsxwriter') as writer:
                    for sheet_name, df in sheets_dict.items():
                        df.to_excel(writer, index=False, sheet_name=sheet_name)
                        workbook = writer.book
                        worksheet = writer.sheets[sheet_name]
                        if sheet_name == SHEET_RECORD and '編號' in df.columns:
                            col_idx = df.columns.get_loc('編號')
                            text_format = workbook.add_format({'num_format': '@'})
                            worksheet.set_column(col_idx, col_idx, None, text_format)
                        if sheet_name == SHEET_COUNT:
                            text_format = workbook.add_format({'num_format': '@'})
                            worksheet.set_column(1, 2, None, text_format)
                        elif sheet_name == SHEET_TYPES:
                            text_format = workbook.add_format({'num_format': '@'})
                            worksheet.set_column(1, 2, None, text_format)
            else:
                # 若無 sheets_dict，至少寫入張數與類型兩表
                result_df = pd.DataFrame(columns=['資料夾', '張數', '廣告圖'])
                type_stats_df = pd.DataFrame(columns=['資料夾', '模特', '平拍'])
                for data_folder_name in st.session_state['folder_values']:
                    num_images = st.session_state['folder_values'][data_folder_name]['張數']
                    ad_images = zfill2_int_str(st.session_state['folder_values'][data_folder_name]['廣告圖'])
                    result_df = pd.concat([result_df, pd.DataFrame([{'資料夾': data_folder_name, '張數': num_images, '廣告圖': ad_images}])], ignore_index=True)

                    model_images = st.session_state['folder_values'][data_folder_name]['模特']
                    flat_images = st.session_state['folder_values'][data_folder_name]['平拍']
                    type_stats_df = pd.concat([type_stats_df, pd.DataFrame([{'資料夾': data_folder_name, '模特': model_images, '平拍': flat_images}])], ignore_index=True)

                with pd.ExcelWriter(excel_bytes_buffer, engine='xlsxwriter') as writer:
                    result_df.to_excel(writer, index=False, sheet_name=SHEET_COUNT)
                    type_stats_df.to_excel(writer, index=False, sheet_name=SHEET_TYPES)
                    workbook = writer.book
                    ws1 = writer.sheets[SHEET_COUNT]
                    ws2 = writer.sheets[SHEET_TYPES]
                    text_format = workbook.add_format({'num_format': '@'})
                    ws1.set_column(1, 2, None, text_format)
                    ws2.set_column(1, 2, None, text_format)

            excel_bytes_buffer.seek(0)
            zipf.writestr(found_excel_name if found_excel_name else f'{EXCEL_KEYWORD}.xlsx', excel_bytes_buffer.getvalue())

        zip_buffer.seek(0)
        return zip_buffer

    def clean_outer_images(zip_buffer):
        """
        產生『已清理』版本的 ZIP：
        - 若專案內沒有 2-IMG，則移除各子包頂層（含 1-Main/2-IMG 同層）的所有圖片檔
        - 保留空目錄
        """
        temp_dir = tempfile.mkdtemp()
        clean_zip_buffer = BytesIO()
        try:
            with zipfile.ZipFile(zip_buffer, "r") as zip_file:
                zip_file.extractall(temp_dir)

            found_2img = any(IMG_DIR in dirs for _, dirs, _ in os.walk(temp_dir))
            if not found_2img:
                for root, dirs, files in os.walk(temp_dir):
                    if TMP_OTHERS_DIRNAME in root.split(os.sep):
                        continue
                    if MAIN_DIR in dirs or IMG_DIR in dirs:
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                                os.remove(file_path)

            with zipfile.ZipFile(clean_zip_buffer, "w", zipfile.ZIP_DEFLATED) as new_zip:
                for root, dirs, files in os.walk(temp_dir):
                    if TMP_OTHERS_DIRNAME in root.split(os.sep):
                        continue
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        if not os.listdir(dir_path):
                            relative_path = os.path.relpath(dir_path, temp_dir)
                            if TMP_OTHERS_DIRNAME not in relative_path.split(os.sep):
                                zip_info = zipfile.ZipInfo(relative_path + "/")
                                new_zip.writestr(zip_info, b"")
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, temp_dir)
                        if TMP_OTHERS_DIRNAME not in relative_path.split(os.sep):
                            write_to_zip(new_zip, file_path, relative_path)
        finally:
            shutil.rmtree(temp_dir)
        clean_zip_buffer.seek(0)
        return clean_zip_buffer

    # =============================================================================
    # 協助函式 - 覆蓋指定路徑並重置 key
    # =============================================================================
    def cover_path_and_reset_key_tab2():
        cover_path_input = st.session_state.get("cover_path_for_sync", "").strip()

        def fast_copy_filtered(src, dst):
            file_list = []
            for root, dirs, files in os.walk(src):
                rel_path = os.path.relpath(root, src)
                target_dir = os.path.join(dst, rel_path)
                os.makedirs(target_dir, exist_ok=True)
                for file in files:
                    if file.lower().endswith('.gsheet'):
                        continue
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_dir, file)
                    file_list.append((src_file, dst_file))
            with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                futures = [executor.submit(shutil.copy2, src_file, dst_file) for src_file, dst_file in file_list]
                for future in as_completed(futures):
                    future.result()

        if cover_path_input:
            first_level_items = os.listdir(cover_path_input)
            for item in first_level_items:
                item_path = os.path.join(cover_path_input, item)
                if os.path.isdir(item_path):
                    if item != "0-上架資料":
                        try:
                            shutil.rmtree(item_path, onerror=lambda f, p, e: (os.chmod(p, stat.S_IWRITE), f(p)))
                        except Exception:
                            pass
                elif os.path.isfile(item_path):
                    if not item.lower().endswith('.gsheet'):
                        try:
                            os.chmod(item_path, stat.S_IWRITE)
                            os.remove(item_path)
                        except PermissionError:
                            try:
                                if os.name == 'nt':
                                    ctypes.windll.kernel32.SetFileAttributesW(item_path, 0x80)
                                os.remove(item_path)
                            except Exception:
                                pass
                        except Exception:
                            pass

            time.sleep(1)
            if "uncleaned_zip_content" in st.session_state and st.session_state["uncleaned_zip_content"]:
                final_zip_bytes = st.session_state["uncleaned_zip_content"]  # 使用未清理版本覆蓋
                tmp_extract_dir = os.path.join(tempfile.gettempdir(), "cover_tmp")
                if os.path.exists(tmp_extract_dir):
                    shutil.rmtree(tmp_extract_dir, ignore_errors=True)
                os.makedirs(tmp_extract_dir, exist_ok=True)
                with zipfile.ZipFile(BytesIO(final_zip_bytes), 'r') as final_zip:
                    final_zip.extractall(tmp_extract_dir)
                tmp_others_extracted = os.path.join(tmp_extract_dir, TMP_OTHERS_DIRNAME)
                if os.path.exists(tmp_others_extracted):
                    shutil.rmtree(tmp_others_extracted, ignore_errors=True)

                fast_copy_filtered(tmp_extract_dir, cover_path_input)
                shutil.rmtree(tmp_extract_dir, ignore_errors=True)

            if "tmp_dir" in st.session_state and os.path.exists(st.session_state["tmp_dir"]):
                shutil.rmtree(st.session_state["tmp_dir"], ignore_errors=True)

        # 重置 key 與狀態
        st.session_state['file_uploader_key2'] += 1
        st.session_state['text_area_key2'] += 1
        st.session_state['file_uploader_disabled_2'] = False
        st.session_state['text_area_disabled_2'] = False
        st.session_state['filename_changes'].clear()

    # =============================================================================
    # 全域暫存資料夾
    # =============================================================================
    cache_base_dir = os.path.join(tempfile.gettempdir(), "streamlit_cache")
    psd_cache_dir = os.path.join(cache_base_dir, "psd_cache")
    ai_cache_dir = os.path.join(cache_base_dir, "ai_cache")
    custom_tmp_dir = os.path.join(cache_base_dir, "custom_tmpdir")

    # =============================================================================
    # 介面初始化與輸入
    # =============================================================================
    initialize_tab2()
    st.write("\n")
    col1, col2 = st.columns(2, vertical_alignment="top")
    uploaded_file_2 = col1.file_uploader(
        "上傳編圖結果 ZIP 檔",
        type=["zip"],
        key='file_uploader_' + str(st.session_state['file_uploader_key2']),
        disabled=st.session_state['file_uploader_disabled_2'],
        on_change=handle_file_uploader_change_tab2,
        label_visibility="collapsed",
    )
    input_path_2 = col2.text_area(
        "   輸入資料夾路徑",
        height=68,
        key='text_area_' + str(st.session_state['text_area_key2']),
        disabled=st.session_state['text_area_disabled_2'],
        on_change=handle_text_area_change_tab2,
        placeholder="  輸入分包資料夾路徑",
        label_visibility="collapsed"
    )

    # =============================================================================
    # 輸入來源載入
    # =============================================================================
    if uploaded_file_2 or input_path_2:
        extract_dir = st.session_state["custom_tmpdir"]
        if not st.session_state.get("source_loaded", False):
            if uploaded_file_2:
                with zipfile.ZipFile(uploaded_file_2) as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif input_path_2:
                # 支援 Windows search-ms 超連結貼上
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
                    copytree_multithreaded(input_path_2, extract_dir)
            st.session_state["source_loaded"] = True

        # 讀取 Excel 與表單
        excel_file_path, found_excel_name = find_excel_in_dir(extract_dir)
        sheets_dict = read_sheets_dict(excel_file_path)
        if SHEET_COUNT in sheets_dict:
            count_sheet_df = sheets_dict[SHEET_COUNT]
            folder_row_index_map = {str(row['資料夾']): idx for idx, row in count_sheet_df.iterrows()}
        else:
            count_sheet_df = pd.DataFrame(columns=['資料夾', '張數', '廣告圖'])
            folder_row_index_map = {}

        if SHEET_TYPES in sheets_dict:
            type_stats_df = sheets_dict[SHEET_TYPES]
        else:
            type_stats_df = pd.DataFrame(columns=['資料夾', '模特', '平拍'])

        # 建立各子包統計預設
        folder_stats_map = {}
        top_level_folders = [
            name for name in os.listdir(extract_dir)
            if os.path.isdir(os.path.join(extract_dir, name))
            and not name.startswith(('_', '.'))
            and name != TMP_OTHERS_DIRNAME
        ]
        for folder_name in top_level_folders:
            matched = False
            for data_folder_name in folder_row_index_map.keys():
                if data_folder_name in folder_name:
                    idx = folder_row_index_map[data_folder_name]
                    row = count_sheet_df.loc[idx]
                    if data_folder_name in type_stats_df['資料夾'].values:
                        type_idx = type_stats_df[type_stats_df['資料夾'] == data_folder_name].index[0]
                        type_row = type_stats_df.loc[type_idx]
                        folder_stats_map[folder_name] = {
                            '資料夾': data_folder_name,
                            '張數': str(row['張數']),
                            '廣告圖': str(row['廣告圖']),
                            '模特': str(type_row['模特']),
                            '平拍': str(type_row['平拍']),
                        }
                    else:
                        folder_stats_map[folder_name] = {
                            '資料夾': data_folder_name,
                            '張數': str(row['張數']),
                            '廣告圖': str(row['廣告圖']),
                            '模特': '0',
                            '平拍': '0',
                        }
                    matched = True
                    break
            if not matched:
                folder_stats_map[folder_name] = {
                    '資料夾': folder_name,
                    '張數': '0',
                    '廣告圖': '1',
                    '模特': '0',
                    '平拍': '0',
                }

        # 將統計值寫進 session_state['folder_values']（首次）
        for folder_name, data in folder_stats_map.items():
            data_folder_name = data.get('資料夾', folder_name)
            if data_folder_name not in st.session_state['folder_values']:
                st.session_state['folder_values'][data_folder_name] = {
                    '張數': data.get('張數', '0'),
                    '廣告圖': data.get('廣告圖', '1'),
                    '模特': data.get('模特', '0'),
                    '平拍': data.get('平拍', '0'),
                }

        # 預先載入所有子包圖片（快取）
        for folder in top_level_folders:
            st.session_state['image_cache'].setdefault(folder, {})
            img_folder_path = os.path.join(extract_dir, folder, IMG_DIR)
            is_main_all = True
            if not os.path.exists(img_folder_path):
                img_folder_path = os.path.join(extract_dir, folder, MAIN_DIR, "All")
                is_main_all = False

            tasks = []
            with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                if os.path.exists(img_folder_path):
                    image_files = get_outer_folder_images(img_folder_path)
                    for image_file in image_files:
                        image_path = os.path.join(img_folder_path, image_file)
                        if image_path not in st.session_state['image_cache'][folder]:
                            add_label = image_file.lower().endswith(LABEL_EXTENSIONS)
                            future = executor.submit(load_and_process_image, image_path, add_label)
                            tasks.append((future, folder, image_path))

                outer_folder_path = os.path.join(extract_dir, folder)
                outer_images = get_outer_folder_images(outer_folder_path)
                for outer_image_file in outer_images:
                    image_path = os.path.join(outer_folder_path, outer_image_file)
                    if image_path not in st.session_state['image_cache'][folder]:
                        add_label = outer_image_file.lower().endswith(LABEL_EXTENSIONS)
                        future = executor.submit(load_and_process_image, image_path, add_label)
                        tasks.append((future, folder, image_path))

                for future, folder_name_, image_path in tasks:
                    try:
                        image = future.result()
                        st.session_state['image_cache'][folder_name_][image_path] = image
                    except Exception as e:
                        st.warning(f"載入圖片 {image_path} 時發生錯誤: {str(e)}")

        # UI：選擇資料夾
        if 'previous_selected_folder' not in st.session_state and top_level_folders:
            st.session_state['previous_selected_folder'] = top_level_folders[0]

        if top_level_folders:
            if 'previous_selected_folder' not in st.session_state:
                st.session_state['previous_selected_folder'] = None
            if 'last_text_inputs' not in st.session_state:
                st.session_state['last_text_inputs'] = {}

            previous_folder = st.session_state['previous_selected_folder']
            # 注意：保留 pills 觸發行為與原先一致
            selected_folder = st.pills(
                "選擇一個資料夾",
                top_level_folders,
                default=top_level_folders[0],
                label_visibility="collapsed",
                on_change=lambda: st.session_state.update({'has_duplicates': False})
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
            if selected_folder is None:
                st.stop()

            img_folder_path = os.path.join(extract_dir, selected_folder, IMG_DIR)
            is_main_all = os.path.exists(img_folder_path)
            if not is_main_all:
                img_folder_path = os.path.join(extract_dir, selected_folder, MAIN_DIR, "All")

            outer_folder_path = os.path.join(extract_dir, selected_folder)
            outer_images = get_outer_folder_images(outer_folder_path)

            if os.path.exists(img_folder_path):
                image_files = get_outer_folder_images(img_folder_path)
                if image_files:
                    st.session_state.setdefault('filename_changes', {})
                    st.session_state.setdefault('confirmed_changes', {})
                    st.session_state.setdefault('image_cache', {})

                    if selected_folder not in st.session_state['filename_changes']:
                        st.session_state['filename_changes'][selected_folder] = {}
                    if selected_folder not in st.session_state['confirmed_changes']:
                        st.session_state['confirmed_changes'][selected_folder] = False
                    if selected_folder not in st.session_state['image_cache']:
                        st.session_state['image_cache'][selected_folder] = {}

                    all_images = set(image_files + outer_images)
                    desired_images, excluded_images = [], []
                    for image_file in all_images:
                        if (selected_folder in st.session_state['filename_changes']
                                and image_file in st.session_state['filename_changes'][selected_folder]):
                            data = st.session_state['filename_changes'][selected_folder][image_file]
                            (excluded_images if data['new_filename'] == '' else desired_images).append(image_file)
                        else:
                            (desired_images if image_file in image_files else excluded_images).append(image_file)

                    desired_images.sort(key=get_sort_key)
                    excluded_images.sort(key=get_sort_key)

                    st.session_state.setdefault('outer_images_map', {})
                    st.session_state['outer_images_map'][selected_folder] = excluded_images

                    basename_to_extensions = defaultdict(list)
                    for image_file in (all_images):
                        basename, ext = os.path.splitext(image_file)
                        basename_to_extensions[basename].append(ext.lower())

                    # 表單（保留 enter_to_submit 與欄位命名）
                    with st.form(f"filename_form_{selected_folder}", enter_to_submit=True):
                        colAA, colBB, colCC = st.columns([19, 0.01, 1.8])
                        with colAA:
                            cols = st.columns(7)
                            for idx, image_file in enumerate(desired_images):
                                if idx % 7 == 0 and idx != 0:
                                    cols = st.columns(7)
                                col = cols[idx % 7]
                                image_path = (os.path.join(img_folder_path, image_file)
                                              if image_file in image_files
                                              else os.path.join(outer_folder_path, image_file))
                                if image_path not in st.session_state['image_cache'][selected_folder]:
                                    add_label = image_file.lower().endswith(LABEL_EXTENSIONS)
                                    image = load_and_process_image(image_path, add_label)
                                    st.session_state['image_cache'][selected_folder][image_path] = image
                                else:
                                    image = st.session_state['image_cache'][selected_folder][image_path]
                                col.image(image, use_container_width=True)

                                filename_without_ext = os.path.splitext(image_file)[0]
                                current_filename = (filename_without_ext if is_main_all
                                                    else (filename_without_ext.split('_', 1)[-1] if '_' in filename_without_ext else filename_without_ext))
                                if (selected_folder in st.session_state['filename_changes']
                                        and image_file in st.session_state['filename_changes'][selected_folder]):
                                    modified_text = st.session_state['filename_changes'][selected_folder][image_file]['text']
                                else:
                                    modified_text = current_filename

                                text_input_key = f"{selected_folder}_{image_file}"
                                if text_input_key not in st.session_state:
                                    st.session_state[text_input_key] = modified_text
                                col.text_input('檔名', key=text_input_key, label_visibility="collapsed")

                            # 右側統計選單
                            if folder_stats_map:
                                data = folder_stats_map.get(selected_folder, {})
                                data_folder_name = data.get('資料夾', selected_folder)
                                if (data_folder_name and 'folder_values' in st.session_state
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

                                num_images_state_key = f"{selected_folder}_num_images"
                                ad_images_state_key = f"{selected_folder}_ad_images"
                                model_count_state_key = f"{selected_folder}_model_images"
                                flat_count_state_key = f"{selected_folder}_flat_images"

                                if num_images_state_key not in st.session_state:
                                    st.session_state[num_images_state_key] = num_images_default
                                if ad_images_state_key not in st.session_state:
                                    st.session_state[ad_images_state_key] = ad_images_default
                                if model_count_state_key not in st.session_state:
                                    st.session_state[model_count_state_key] = model_images_default
                                if flat_count_state_key not in st.session_state:
                                    st.session_state[flat_count_state_key] = flat_images_default

                                upper_limit = len(desired_images) + len(excluded_images)
                                num_images_options = [str(i) for i in range(0, upper_limit + 1)]
                                ad_images_options = [str(i) for i in range(1, upper_limit + 1)]
                                type_images_options = [str(i) for i in range(0, upper_limit + 1)]
                                with colCC:
                                    st.selectbox('張數', num_images_options, key=num_images_state_key)
                                    st.selectbox('廣告圖', ad_images_options, key=ad_images_state_key)
                                    st.selectbox('模特數', type_images_options, key=model_count_state_key)
                                    st.selectbox('平拍數', type_images_options, key=flat_count_state_key)
                            else:
                                num_images_state_key = None
                                ad_images_state_key = None
                                folder_stats_map = None

                        st.divider()
                        colA, colB, colC, colD = st.columns([3, 5, 8, 2.5], vertical_alignment="center")

                        # 暫存修改（依資料結構選對函式）
                        if colA.form_submit_button(
                            "暫存修改",
                            on_click=submit_img_folder if is_main_all else submit_main_folder,
                            args=(selected_folder, desired_images, excluded_images, folder_stats_map)
                        ):
                            if st.session_state.get('has_duplicates') is False:
                                st.toast(f"資料夾 {selected_folder} 暫存修改成功!", icon='🎉')

                        # 外層圖片（只在 1-Main/All 顯示）
                        if excluded_images and not is_main_all:
                            with colD.popover("外層圖片"):
                                outer_cols = st.columns(6)
                                for idx, outer_image_file in enumerate(excluded_images):
                                    if idx % 6 == 0 and idx != 0:
                                        outer_cols = st.columns(6)
                                    col = outer_cols[idx % 6]
                                    outer_image_path = (os.path.join(outer_folder_path, outer_image_file)
                                                        if outer_image_file in outer_images
                                                        else os.path.join(img_folder_path, outer_image_file))
                                    if outer_image_path not in st.session_state['image_cache'][selected_folder]:
                                        add_label = outer_image_file.lower().endswith(LABEL_EXTENSIONS)
                                        outer_image = load_and_process_image(outer_image_path, add_label)
                                        st.session_state['image_cache'][selected_folder][outer_image_path] = outer_image
                                    else:
                                        outer_image = st.session_state['image_cache'][selected_folder][outer_image_path]
                                    col.image(outer_image, use_container_width=True)

                                    current_filename = os.path.splitext(outer_image_file)[0]
                                    if (selected_folder in st.session_state['filename_changes']
                                            and outer_image_file in st.session_state['filename_changes'][selected_folder]):
                                        modified_text = st.session_state['filename_changes'][selected_folder][outer_image_file]['text']
                                        if modified_text == '':
                                            modified_text = st.session_state['filename_changes'][selected_folder][outer_image_file].get('last_non_empty', current_filename)
                                    else:
                                        modified_text = current_filename
                                    text_input_key = f"outer_{selected_folder}_{outer_image_file}"
                                    col.text_input('檔名', value=modified_text, key=text_input_key)

                        if st.session_state.get('has_duplicates'):
                            colB.warning(f"檔名重複: {', '.join(st.session_state['duplicate_filenames'])}")

                    # ---- 全部確認與產出 ----
                    if st.checkbox("所有資料夾均確認完成"):
                        if not st.session_state.get('processing_complete', False):
                            with st.spinner('檔案處理中...'):
                                tmp_dir_for_others = os.path.join(extract_dir, TMP_OTHERS_DIRNAME)
                                st.session_state["tmp_dir"] = tmp_dir_for_others
                                setup_temporary_directory(extract_dir, tmp_dir_for_others)

                                # 1) 產生「未清理」zip
                                uncleaned_zip_buffer = parallel_read_zip_compress()
                                st.session_state['uncleaned_zip_content'] = uncleaned_zip_buffer.getvalue()

                                # 2) 產生「已清理」zip（刪除不該留在外層的圖片）
                                cleaned_zip_buffer = clean_outer_images(uncleaned_zip_buffer)
                                st.session_state['cleaned_zip_content'] = cleaned_zip_buffer.getvalue()

                                # 決定下載檔名
                                if uploaded_file_2:
                                    download_file_name = uploaded_file_2.name.replace(".zip", "_已複檢.zip")
                                elif input_path_2:
                                    folder_name = os.path.basename(input_path_2.strip(os.sep))
                                    download_file_name = f"{folder_name}__已複檢.zip"
                                else:
                                    download_file_name = "結果_已複檢.zip"
                                st.session_state["download_file_name"] = download_file_name

                                # 標記完成並 rerun
                                st.session_state['processing_complete'] = True
                                st.rerun()

                        # 顯示覆蓋與下載
                        st.write("\n")
                        col1_, col2_, col3_ = st.columns([4, 0.1, 2], vertical_alignment="center")

                        if not uploaded_file_2 and input_path_2:
                            cover_text_default = input_path_2.strip()
                        elif st.session_state.get("input_path_from_tab1"):
                            cover_text_default = st.session_state.get("input_path_from_tab1")
                        else:
                            cover_text_default = ""

                        col1_.text_input(
                            label="同步覆蓋此路徑的檔案",
                            placeholder="同步覆蓋此路徑的檔案",
                            value=cover_text_default,
                            key="cover_path_for_sync"
                        )

                        cleaned_zip_for_download = st.session_state.get("cleaned_zip_content")
                        download_name = st.session_state.get("download_file_name", "結果_已複檢.zip")

                        if cleaned_zip_for_download:
                            col3_.download_button(
                                label='下載 + 覆蓋(選填)',
                                data=cleaned_zip_for_download,   # 下載使用「已清理」版本
                                file_name=download_name,
                                mime='application/zip',
                                on_click=cover_path_and_reset_key_tab2
                            )
                else:
                    st.error("未找到圖片。")
            else:
                st.error("不存在 '2-IMG' 或 '1-Main/All' 資料夾。")
        else:
            st.error("未找到任何資料夾。")
