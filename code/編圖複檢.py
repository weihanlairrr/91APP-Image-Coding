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
import subprocess
import sys
from io import BytesIO
from psd_tools import PSDImage
from collections import Counter, defaultdict
from PIL import Image, ImageOps, ImageDraw, ImageFont, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz

def tab2():
    ############################################################################
    # 1. 初始設定與 Session State 管理
    ############################################################################
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

    ############################################################################
    # 2. 臨時目錄與 ZIP 檔處理工具
    ############################################################################
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
                if is_same_level_as_image_folders and ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.psd', ".ai"]:
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
                write_to_zip(zipf, file_path, relative_path)
            # 處理空資料夾
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):  # 空資料夾
                    relative_path = os.path.relpath(dir_path, tmp_dir)
                    zip_info = zipfile.ZipInfo(relative_path + "/")
                    zipf.writestr(zip_info, b"")

    ############################################################################
    # 3. 圖片處理工具
    ############################################################################
    def get_outer_folder_images(folder_path):
        """
        獲取指定資料夾中所有圖片檔案，並按名稱排序。
        支援 jpg、jpeg、png、tif、psd、ai
        """
        return sorted(
            [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff', 'psd', 'ai'))
            ]
        )

    def get_prefix(image_files):
        """
        從圖片檔案中取得通用的命名前綴。
        :param image_files: 圖片檔案列表
        :return: 圖片檔名的前綴字串（若找不到則回傳空字串）
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
        """
        加載並處理圖片，支持 PSD 格式與 AI 格式。
        透過 lazy 模式解析 PSD 內嵌 Composite 並快取為 JPG。
        """
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.psd':
            cache_dir = os.path.join(tempfile.gettempdir(), "psd_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file_name = str(abs(hash(image_path))) + ".jpg"
            cache_path = os.path.join(cache_dir, cache_file_name)
            if os.path.exists(cache_path):
                image = Image.open(cache_path).convert('RGB')
            else:
                psd = PSDImage.open(image_path, lazy=True)
                image = psd.composite(force=False)
                if image:
                    image = image.convert('RGB')
                    image.save(cache_path, format='JPEG', quality=80)
                else:
                    raise Exception("無法處理 PSD 文件")
        elif ext == '.ai':
            # 新增 AI 快取機制，將 DPI 從 150 調低至 100
            cache_dir = os.path.join(tempfile.gettempdir(), "ai_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file_name = str(abs(hash(image_path))) + ".jpg"
            cache_path = os.path.join(cache_dir, cache_file_name)
            if os.path.exists(cache_path):
                image = Image.open(cache_path).convert('RGB')
            else:
                try:
                    doc = fitz.open(image_path)
                    page = doc.load_page(0)
                    pix = page.get_pixmap(dpi=150)  # DPI 由 150 降為 100
                    image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
                    doc.close()
                    image.save(cache_path, format='JPEG')
                except Exception as e:
                    raise Exception(f"無法處理 .ai 檔案: {str(e)}")
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except UnidentifiedImageError:
                with open(image_path, 'rb') as f:
                    raw_data = f.read()
                    decoded_data = imagecodecs.tiff_decode(raw_data)
                    image = Image.fromarray(decoded_data).convert('RGB')
        
        image = ImageOps.pad(image, (1000, 1000), method=Image.Resampling.LANCZOS, color=(255, 255, 255))
        if add_label:
            image = add_image_label(image, ext)
        return image

    def get_sort_key(image_file):
        """
        取得排序用的 key：
        若檔名中包含數字，則以數字大小排序，否則採用字母順序。
        若在 filename_changes 中有新檔名，則使用新檔名排序。
        """
        filename_changes = st.session_state.get('filename_changes', {}).get(selected_folder, {})
        if image_file in filename_changes:
            new_filename = filename_changes[image_file]['new_filename']
            filename = new_filename if new_filename else image_file
        else:
            filename = image_file
        match = re.search(r'(\d+)', filename)
        if match:
            num = int(match.group(1))
            return (0, num, filename)
        else:
            return (1, filename)

    ############################################################################
    # 4. UI 事件處理函式
    ############################################################################
    def handle_file_uploader_change_tab2():
        """
        檔案上傳變更時的處理邏輯：檢查是否換檔並清空相關暫存。
        """
        file_key = 'file_uploader_' + str(st.session_state.get('file_uploader_key2', 0))
        uploaded_file_1 = st.session_state.get(file_key, None)
        if uploaded_file_1:
            current_filename = uploaded_file_1.name
            if current_filename != st.session_state['previous_uploaded_file_name']:
                if "custom_tmpdir" in st.session_state and st.session_state["custom_tmpdir"]:
                    shutil.rmtree(st.session_state["custom_tmpdir"], ignore_errors=True)
                st.session_state["custom_tmpdir"] = tempfile.mkdtemp()
                st.session_state['image_cache'].clear()
                st.session_state['filename_changes'].clear()
                st.session_state['confirmed_changes'].clear()
                st.session_state['folder_values'].clear()
                st.session_state['previous_uploaded_file_name'] = current_filename
        st.session_state.text_area_disabled_2 = bool(uploaded_file_1)

    def handle_text_area_change_tab2():
        """
        處理路徑輸入變更的邏輯：檢查是否換路徑並清空相關暫存。
        """
        text_key = 'text_area_' + str(st.session_state.get('text_area_key2', 0))
        text_content = st.session_state.get(text_key, "").strip()
        if text_content != st.session_state['previous_input_path']:
            if "custom_tmpdir" in st.session_state and st.session_state["custom_tmpdir"]:
                shutil.rmtree(st.session_state["custom_tmpdir"], ignore_errors=True)
            st.session_state["custom_tmpdir"] = tempfile.mkdtemp()
            st.session_state['image_cache'].clear()
            st.session_state['filename_changes'].clear()
            st.session_state['confirmed_changes'].clear()
            st.session_state['folder_values'].clear()
            st.session_state['previous_input_path'] = text_content
        st.session_state.file_uploader_disabled_2 = bool(text_content)

    ############################################################################
    # 5. 檔名修改與提交處理函式
    ############################################################################
    def handle_submission(selected_folder, images_to_display, outer_images_to_display, use_full_filename, folder_to_data):
        """
        處理圖片檔名修改的提交邏輯，包含重命名與重複檢查。
        """
        current_filenames = {}
        temp_filename_changes = {}
        modified_outer_count = 0
        removed_image_count = 0
        if not use_full_filename:
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
            if new_text.strip() != default_text:
                temp_filename_changes[outer_image_file] = {
                    'new_filename': new_filename,
                    'text': new_text
                }
                if new_filename != '':
                    modified_outer_count += 1
        new_filenames = [
            data['new_filename']
            for data in temp_filename_changes.values()
            if data['new_filename'] != ''
        ]
        duplicates = [
            filename for filename, count in Counter(new_filenames).items() if count > 1
        ]
        if duplicates:
            st.session_state['has_duplicates'] = True
            st.session_state['duplicate_filenames'] = duplicates
            st.session_state['confirmed_changes'][selected_folder] = False
        else:
            st.session_state['has_duplicates'] = False
            st.session_state['confirmed_changes'][selected_folder] = True
            if not use_full_filename:
                def sort_key(item):
                    text = item[1]['text'].strip()
                    if text.isdigit():
                        return (0, int(text))
                    return (1, text)
                sorted_files = sorted(
                    ((file, data) for file, data in temp_filename_changes.items() if data['new_filename'] != ''),
                    key=sort_key
                )
                rename_counter = 1
                for file, data in sorted_files:
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
            st.session_state[num_images_key] = str(
                max(1, current_num_images - removed_image_count + modified_outer_count)
            )
        ad_images_key = f"{selected_folder}_ad_images"
        ad_images_value = st.session_state.get(ad_images_key)
        model_images_key = f"{selected_folder}_model_images"
        flat_images_key = f"{selected_folder}_flat_images"
        model_images_value = st.session_state.get(model_images_key)
        flat_images_value = st.session_state.get(flat_images_key)
        data = folder_to_data.get(selected_folder, {})
        data_folder_name = data.get('資料夾', selected_folder)
        st.session_state['folder_values'][data_folder_name] = {
            '張數': st.session_state[num_images_key],
            '廣告圖': ad_images_value,
            '模特': model_images_value,
            '平拍': flat_images_value,
        }
        st.session_state['modified_folders'].add(data_folder_name)

    ############################################################################
    # 6. 壓縮與 ZIP 寫入工具
    ############################################################################
    def get_compress_type(file_path):
        """
        若檔案為圖片類型（已壓縮格式），則採用 ZIP_STORED，否則使用 ZIP_DEFLATED。
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.psd', '.ai']:
            return zipfile.ZIP_STORED
        else:
            return zipfile.ZIP_DEFLATED

    def write_to_zip(zipf, file_path, arcname):
        """
        根據檔案副檔名決定使用的壓縮方式，對 ZIP_DEFLATED 採用較低 compresslevel 以加速壓縮。
        """
        ct = get_compress_type(file_path)
        if ct == zipfile.ZIP_DEFLATED:
            zipf.write(file_path, arcname, compress_type=ct, compresslevel=1)
        else:
            zipf.write(file_path, arcname, compress_type=ct)

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
            for root, dirs, files in os.walk(temp_dir):
                if "tmp_others" in root.split(os.sep):
                    continue
                if "1-Main" in dirs or "2-IMG" in dirs:
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                            os.remove(file_path)
            with zipfile.ZipFile(cleaned_zip_buffer, "w", zipfile.ZIP_DEFLATED) as new_zip:
                for root, dirs, files in os.walk(temp_dir):
                    if "tmp_others" in root.split(os.sep):
                        continue
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        if not os.listdir(dir_path):
                            relative_path = os.path.relpath(dir_path, temp_dir)
                            if "tmp_others" not in relative_path.split(os.sep):
                                zip_info = zipfile.ZipInfo(relative_path + "/")
                                new_zip.writestr(zip_info, b"")
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, temp_dir)
                        if "tmp_others" not in relative_path.split(os.sep):
                            write_to_zip(new_zip, file_path, relative_path)
        finally:
            shutil.rmtree(temp_dir)
        cleaned_zip_buffer.seek(0)
        return cleaned_zip_buffer

    ############################################################################
    # 7. 最終覆蓋路徑與重置上傳器狀態
    ############################################################################
    def cover_path_and_reset_key_tab2():
        """
        重置文件上傳器狀態，並用最終 ZIP 檔案內容覆蓋指定路徑，
        同時處理可能無法刪除的 .db 檔案（透過終止相關進程強制刪除）。
        """
        if cover_path_input.strip():
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
                        if file.lower() == '編圖結果.xlsx':
                            os.remove(file_path)
                        elif ext not in [".xlsx", ".gsheet", ".ai"]:
                            try:
                                os.remove(file_path)
                            except PermissionError:
                                try:
                                    if os.name == 'nt':
                                        command = f'handle.exe "{file_path}"'
                                        output = subprocess.check_output(command, shell=True, text=True)
                                        for line in output.splitlines():
                                            if "pid:" in line.lower():
                                                pid = int(line.split("pid:")[1].split()[0])
                                                os.system(f"taskkill /PID {pid} /F")
                                    else:
                                        command = f'lsof | grep "{file_path}"'
                                        output = subprocess.check_output(command, shell=True, text=True)
                                        for line in output.splitlines():
                                            pid = int(line.split()[1])
                                            os.kill(pid, 9)
                                    os.remove(file_path)
                                except Exception as e:
                                    st.warning(f"無法刪除檔案: {file_path}，錯誤: {str(e)}")
                    except PermissionError:
                        try:
                            if os.name == 'nt':
                                ctypes.windll.kernel32.SetFileAttributesW(file_path, 0x80)
                                os.remove(file_path)
                        except Exception as e:
                            st.warning(f"無法刪除檔案: {file_path}，錯誤: {str(e)}")
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
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

    ############################################################################
    # 8. 主流程：檔案來源處理、圖片與 Excel 載入、介面建立與檔案打包下載
    ############################################################################
    initialize_tab2()
    st.write("\n")
    col1, col2 = st.columns(2, vertical_alignment="top")

    uploaded_file_2 = col1.file_uploader(
        "上傳編圖結果 ZIP 檔",
        type=["zip"],
        key='file_uploader_' + str(st.session_state['file_uploader_key2']),
        disabled=st.session_state['file_uploader_disabled_2'],
        on_change=handle_file_uploader_change_tab2,
        label_visibility="collapsed"
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
        # 將頂層資料夾對應到 Excel 內的統計資料
        folder_to_data = {}
        top_level_folders = [
            name for name in os.listdir(tmpdirname)
            if os.path.isdir(os.path.join(tmpdirname, name))
            and not name.startswith(('_', '.'))
            and name != "tmp_others"
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
        # 初始化 session_state 中的 folder_values
        for folder_name, data in folder_to_data.items():
            data_folder_name = data.get('資料夾', folder_name)
            if data_folder_name not in st.session_state['folder_values']:
                st.session_state['folder_values'][data_folder_name] = {
                    '張數': data.get('張數', '1'),
                    '廣告圖': data.get('廣告圖', '1'),
                    '模特': data.get('模特', '0'),
                    '平拍': data.get('平拍', '0'),
                }
        # 上傳檔案時預先載入所有圖片快取（多執行緒）
        for folder in top_level_folders:
            st.session_state['image_cache'].setdefault(folder, {})
            img_folder_path = os.path.join(tmpdirname, folder, '2-IMG')
            use_full_filename = True
            if not os.path.exists(img_folder_path):
                img_folder_path = os.path.join(tmpdirname, folder, '1-Main', 'All')
                use_full_filename = False
            tasks = []
            with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                if os.path.exists(img_folder_path):
                    image_files = get_outer_folder_images(img_folder_path)
                    for image_file in image_files:
                        image_path = os.path.join(img_folder_path, image_file)
                        if image_path not in st.session_state['image_cache'][folder]:
                            add_label = (image_file.lower().endswith('.png')
                                         or image_file.lower().endswith('.tif')
                                         or image_file.lower().endswith('.tiff')
                                         or image_file.lower().endswith('.psd')
                                         or image_file.lower().endswith('.ai'))
                            future = executor.submit(load_and_process_image, image_path, add_label)
                            tasks.append((future, folder, image_path))
                outer_folder_path = os.path.join(tmpdirname, folder)
                outer_images = get_outer_folder_images(outer_folder_path)
                for outer_image_file in outer_images:
                    image_path = os.path.join(outer_folder_path, outer_image_file)
                    if image_path not in st.session_state['image_cache'][folder]:
                        add_label = (outer_image_file.lower().endswith('.png')
                                     or outer_image_file.lower().endswith('.tif')
                                     or outer_image_file.lower().endswith('.tiff')
                                     or outer_image_file.lower().endswith('.psd')
                                     or outer_image_file.lower().endswith('.ai'))
                        future = executor.submit(load_and_process_image, image_path, add_label)
                        tasks.append((future, folder, image_path))
                for future, folder, image_path in tasks:
                    try:
                        image = future.result()
                        st.session_state['image_cache'][folder][image_path] = image
                    except Exception as e:
                        st.warning(f"載入圖片 {image_path} 時發生錯誤: {str(e)}")
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
                                        or image_file.lower().endswith('.ai')
                                    )
                                    image = load_and_process_image(image_path, add_label)
                                    st.session_state['image_cache'][selected_folder][image_path] = image
                                else:
                                    image = st.session_state['image_cache'][selected_folder][image_path]
                                col.image(image, use_container_width=True)
                                filename_without_ext = os.path.splitext(image_file)[0]
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
                                upper_limit = max(10, int(num_images_default), int(ad_images_default))
                                num_images_options = [str(i) for i in range(1, upper_limit + 1)]
                                ad_images_options = [str(i) for i in range(1, upper_limit + 1)]
                                type_images_options = [str(i) for i in range(0, 11)]
                                with colCC:
                                    st.selectbox('張數', num_images_options, key=num_images_key)
                                    st.selectbox('廣告圖', ad_images_options, key=ad_images_key)
                                    st.selectbox('模特數', type_images_options, key=model_images_key)
                                    st.selectbox('平拍', type_images_options, key=flat_images_key)
                            else:
                                num_images_key = None
                                ad_images_key = None
                                folder_to_data = None
                        st.divider()
                        colA, colB, colC, colD = st.columns([3, 7, 2, 2.5], vertical_alignment="center")
                        if colA.form_submit_button(
                            "暫存修改",
                            on_click=handle_submission,
                            args=(selected_folder, images_to_display, outer_images_to_display, use_full_filename, folder_to_data)
                        ):
                            if st.session_state.get('has_duplicates') is False:
                                st.toast(f"資料夾 {selected_folder} 暫存修改成功!", icon='🎉')
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
                                    if (outer_image_path not in st.session_state['image_cache'][selected_folder]):
                                        add_label = (
                                            outer_image_file.lower().endswith('.png')
                                            or outer_image_file.lower().endswith('.tif')
                                            or outer_image_file.lower().endswith('.tiff')
                                            or outer_image_file.lower().endswith('.psd')
                                            or outer_image_file.lower().endswith('.ai')
                                        )
                                        outer_image = load_and_process_image(outer_image_path, add_label)
                                        st.session_state['image_cache'][selected_folder][outer_image_path] = outer_image
                                    else:
                                        outer_image = st.session_state['image_cache'][selected_folder][outer_image_path]
                                    col.image(outer_image, use_container_width=True)
                                    filename_without_ext = os.path.splitext(outer_image_file)[0]
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
                                                'last_non_empty', default_text
                                            )
                                    else:
                                        modified_text = default_text
                                    text_input_key = f"outer_{selected_folder}_{outer_image_file}"
                                    col.text_input('檔名', value=modified_text, key=text_input_key)
                        if st.session_state.get('has_duplicates'):
                            colB.warning(f"檔名重複: {', '.join(st.session_state['duplicate_filenames'])}")
                    if st.checkbox("所有資料夾均確認完成"):
                        with st.spinner('檔案處理中...'):
                            tmp_dir_for_others = os.path.join(tmpdirname, "tmp_others")
                            st.session_state["tmp_dir"] = tmp_dir_for_others
                            image_folder = "2-IMG" if os.path.exists(os.path.join(tmpdirname, "2-IMG")) else os.path.join("1-Main", "All")
                            setup_temporary_directory(tmpdirname, tmp_dir_for_others, image_folder)
                            
                            def parallel_read_zip_compress():
                                """
                                先以多執行緒並行讀取所有檔案，再統一寫入 ZIP。
                                """
                                all_files = []
                                top_level_files = [
                                    name for name in os.listdir(tmpdirname)
                                    if os.path.isfile(os.path.join(tmpdirname, name))
                                ]
                                for file_name in top_level_files:
                                    file_path = os.path.join(tmpdirname, file_name)
                                    arcname = file_name
                                    if file_name != '編圖結果.xlsx':
                                        all_files.append((file_path, arcname))
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
                                                    if os.path.exists(os.path.join(tmpdirname, folder_name, '2-IMG')):
                                                        idx = path_parts.index(folder_name)
                                                        path_parts = path_parts[:idx+1] + ['2-IMG', new_filename]
                                                    else:
                                                        idx = path_parts.index(folder_name)
                                                        path_parts = path_parts[:idx+1] + ['1-Main', 'All', new_filename]
                                                    new_rel_path = os.path.join(*path_parts)
                                                all_files.append((full_path, new_rel_path))
                                            else:
                                                all_files.append((full_path, rel_path))
                                file_data_map = {}
                                with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                                    futures = {executor.submit(lambda p: open(p, 'rb').read(), f[0]): f for f in all_files}
                                    for future in as_completed(futures):
                                        fpath, arc = futures[future]
                                        try:
                                            file_data_map[(fpath, arc)] = future.result()
                                        except Exception as e:
                                            st.error(f"讀取檔案時發生錯誤 {fpath}: {str(e)}")
                                zip_buffer = BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                    for (file_path, arcname), data_bytes in file_data_map.items():
                                        ct = get_compress_type(file_path)
                                        info = zipfile.ZipInfo(arcname)
                                        if ct == zipfile.ZIP_DEFLATED:
                                            zipf.writestr(info, data_bytes, compress_type=ct, compresslevel=1)
                                        else:
                                            zipf.writestr(info, data_bytes, compress_type=ct)
                                    merge_temporary_directory_to_zip(zipf, tmp_dir_for_others)
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
                                return zip_buffer
                            zip_buffer = parallel_read_zip_compress()
                            st.session_state["final_zip_content"] = zip_buffer.getvalue()
                            cleaned_zip_buffer = clean_outer_images(zip_buffer)
                            if uploaded_file_2:
                                download_file_name = uploaded_file_2.name.replace(".zip", "_已複檢.zip")
                            elif input_path_2:
                                folder_name = os.path.basename(input_path_2.strip(os.sep))
                                download_file_name = f"{folder_name}__已複檢.zip"
                            else:
                                download_file_name = "結果_已複檢.zip"
                            col1_, col2_ = st.columns([2.7, 1], vertical_alignment="center")
                            if st.session_state.get("input_path_from_tab1"):
                                cover_text_default = st.session_state.get("input_path_from_tab1")
                            elif not uploaded_file_2 and input_path_2:
                                cover_text_default = input_path_2.strip()
                            else:
                                cover_text_default = ""
                            global cover_path_input
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
                else:
                    st.error("未找到圖片。")
            else:
                st.error("不存在 '2-IMG' 或 '1-Main/All' 資料夾。")
        else:
            st.error("未找到任何資料夾。")
