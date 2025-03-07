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
import concurrent.futures
import fitz
from io import BytesIO
from psd_tools import PSDImage
from collections import Counter, defaultdict
from PIL import Image, ImageOps, ImageDraw, ImageFont, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed

def tab2():
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

    def initialize_tab2():
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
            'input_path_from_tab1': "",
            "custom_tmpdir": fixed_custom_tmpdir,
            'previous_selected_folder': None,
            'final_zip_content': None,
            'source_loaded': False,
            'image_original_title': {},
            'image_labels': {}
        }
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)

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

    def merge_temporary_directory_to_zip(zipf, tmp_dir):
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

    def get_outer_folder_images(folder_path):
        return sorted(
            [f for f in os.listdir(folder_path)
             if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'psd', 'ai'))]
        )

    def get_prefix(image_files):
        for image_file in image_files:
            filename_without_ext = os.path.splitext(image_file)[0]
            first_underscore_index = filename_without_ext.find('_')
            if first_underscore_index != -1:
                return filename_without_ext[:first_underscore_index + 1]
        return ""

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
            try:
                image = Image.open(image_path)
            except UnidentifiedImageError:
                with open(image_path, 'rb') as f:
                    raw_data = f.read()
                    decoded_data = imagecodecs.tiff_decode(raw_data)
                    image = Image.fromarray(decoded_data)
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

    def get_sort_key(image_file):
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

    def clear_all_caches():
        shutil.rmtree(fixed_cache_base_dir, ignore_errors=True)
        os.makedirs(fixed_psd_cache_dir, exist_ok=True)
        os.makedirs(fixed_ai_cache_dir, exist_ok=True)
        os.makedirs(fixed_custom_tmpdir, exist_ok=True)

    def handle_file_uploader_change_tab2():
        clear_all_caches()
        st.session_state["custom_tmpdir"] = fixed_custom_tmpdir
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
        clear_all_caches()
        st.session_state["custom_tmpdir"] = fixed_custom_tmpdir
        text_key = 'text_area_' + str(st.session_state.get('text_area_key2', 0))
        text_content = st.session_state.get(text_key, "").strip()
        st.session_state['image_cache'].clear()
        st.session_state['filename_changes'].clear()
        st.session_state['confirmed_changes'].clear()
        st.session_state['folder_values'].clear()
        st.session_state['previous_input_path'] = text_content
        st.session_state["source_loaded"] = False
        st.session_state.file_uploader_disabled_2 = bool(text_content)

    def handle_submission(selected_folder, images_to_display, outer_images_to_display, use_full_filename, folder_to_data):
        removed_image_count = 0
        modified_outer_count = 0
        added_image_count = 0
        removed_model_count = 0
        removed_flat_count = 0
        added_model_count = 0
        added_flat_count = 0
    
        if use_full_filename:
            prefix = ""
        else:
            prefix = get_prefix(images_to_display)
    
        mapping = {}
        excel_file_path = None
        for f in os.listdir(st.session_state["custom_tmpdir"]):
            if f.lower().endswith('.xlsx') and '編圖結果' in f:
                excel_file_path = os.path.join(st.session_state["custom_tmpdir"], f)
                break
        df_result = None
        if excel_file_path and os.path.exists(excel_file_path):
            try:
                sheets = pd.read_excel(excel_file_path, sheet_name=None, dtype={'前綴': str, '編號': str})
                if "編圖紀錄" in sheets:
                    df_result = sheets["編圖紀錄"]
                    tmp_map = {}
                    for idx, row in df_result.iterrows():
                        folder_name = str(row.iloc[0]).strip()
                        original_full_filename = str(row.iloc[1]).strip()
                        orig_filename_only = os.path.basename(original_full_filename.replace("\\", "/"))
                        orig_text = os.path.splitext(orig_filename_only)[0]
                        prefix_val = str(row.iloc[4]).strip()
                        number_val = str(row.iloc[5]).strip()
                        key = (folder_name, prefix_val + number_val)
                        tmp_map[key] = orig_text
                    mapping = tmp_map
                else:
                    df_result = None
            except Exception:
                df_result = None
    
        st.session_state.setdefault('image_original_title', {})
        st.session_state['image_original_title'].setdefault(selected_folder, {})
        for image_file in (images_to_display + outer_images_to_display):
            if image_file not in st.session_state['image_original_title'][selected_folder]:
                norm_folder = selected_folder[:-3] if selected_folder.endswith("_OK") else selected_folder
                default_text = os.path.splitext(os.path.basename(image_file))[0]
                if df_result is not None:
                    key = (norm_folder, default_text)
                    st.session_state['image_original_title'][selected_folder][image_file] = mapping.get(key, default_text)
                else:
                    st.session_state['image_original_title'][selected_folder][image_file] = default_text
    
        def get_image_label(folder, image_file, df):
            norm_folder = folder[:-3] if folder.endswith("_OK") else folder
            df_folder = df[df.iloc[:, 0].astype(str).str.strip() == norm_folder]
            base_name = os.path.splitext(image_file)[0]
            for idx, row_ in df_folder.iterrows():
                prefix_val = str(row_.iloc[4]).strip()
                number_val = str(row_.iloc[5]).strip()
                combined = prefix_val + number_val
                if combined == base_name:
                    description = str(row_.iloc[3])
                    return "模特" if any(keyword in description for keyword in ["模特", "_9", "-0m"]) else "平拍"
            for idx, row_ in df_folder.iterrows():
                original_full_filename = str(row_.iloc[1]).strip()
                orig_filename_only = os.path.basename(original_full_filename.replace("\\", "/"))
                orig_text = os.path.splitext(orig_filename_only)[0]
                if base_name in orig_text:
                    description = str(row_.iloc[3])
                    return "模特" if any(keyword in description for keyword in ["模特", "_9", "-0m"]) else "平拍"
            return "無"
    
        st.session_state.setdefault('image_labels', {})
        st.session_state['image_labels'].setdefault(selected_folder, {})
        for image_file in (images_to_display + outer_images_to_display):
            if image_file not in st.session_state['image_labels'][selected_folder]:
                st.session_state['image_labels'][selected_folder][image_file] = (
                    get_image_label(selected_folder, image_file, df_result) if df_result is not None else "無"
                )
    
        current_filenames = {}
        temp_filename_changes = {}
    
        # 處理主要圖片（來自 2-IMG 或 1-Main/All，但顯示在內層的部分）
        for image_file in images_to_display:
            text_input_key = f"{selected_folder}_{image_file}"
            new_text = st.session_state.get(text_input_key, "")
            extension = os.path.splitext(image_file)[1]
    
            if (selected_folder in st.session_state.get('filename_changes', {}) and 
                image_file in st.session_state['filename_changes'][selected_folder]):
                default_text = st.session_state['filename_changes'][selected_folder][image_file]['text']
            else:
                if use_full_filename:
                    default_text = os.path.splitext(os.path.basename(image_file))[0]
                else:
                    default_text = st.session_state['image_original_title'][selected_folder].get(
                        image_file, os.path.splitext(os.path.basename(image_file))[0]
                    )
    
            if new_text.strip() == '':
                original_title = st.session_state['image_original_title'][selected_folder].get(image_file, "無")
                if (original_title != "無") and (default_text != original_title):
                    if use_full_filename:
                        new_text = original_title
                        new_filename = original_title + extension
                    else:
                        new_text = original_title
                        new_filename = ''  # 內層圖片後續依編號重新命名
                    removed_image_count += 1
                    label_ = st.session_state['image_labels'][selected_folder].get(image_file, "無")
                    if label_ == "模特":
                        removed_model_count += 1
                    elif label_ == "平拍":
                        removed_flat_count += 1
                else:
                    new_text = default_text
                    if use_full_filename:
                        new_filename = default_text + extension
                    else:
                        new_filename = ''
                        removed_image_count += 1
            else:
                if use_full_filename:
                    if new_text.strip() != default_text:
                        original_title = st.session_state['image_original_title'][selected_folder].get(image_file, "無")
                        if default_text == original_title:
                            added_image_count += 1
                            label_ = st.session_state['image_labels'][selected_folder].get(image_file, "無")
                            if label_ == "模特":
                                added_model_count += 1
                            elif label_ == "平拍":
                                added_flat_count += 1
                    new_filename = new_text + extension
                else:
                    new_filename = prefix + new_text + extension
    
            current_filenames[image_file] = {'new_filename': new_filename, 'text': new_text}
            temp_filename_changes[image_file] = {'new_filename': new_filename, 'text': new_text}
    
        # 處理外層圖片（來自 1-Main/All，需移到 popover 裡面）
        outer_moved_count = 0
        for outer_image_file in outer_images_to_display:
            text_input_key = f"outer_{selected_folder}_{outer_image_file}"
            new_text = st.session_state.get(text_input_key, "")
            extension = os.path.splitext(outer_image_file)[1]
    
            if (selected_folder in st.session_state.get('filename_changes', {}) and 
                outer_image_file in st.session_state['filename_changes'][selected_folder]):
                default_text = st.session_state['filename_changes'][selected_folder][outer_image_file]['text']
            else:
                if use_full_filename:
                    default_text = os.path.splitext(os.path.basename(outer_image_file))[0]
                else:
                    default_text = st.session_state['image_original_title'][selected_folder].get(
                        outer_image_file, os.path.splitext(os.path.basename(outer_image_file))[0]
                    )
    
            if new_text.strip() == '':
                original_title = st.session_state['image_original_title'][selected_folder].get(outer_image_file, "無")
                # 不論 full mode 與否，外層圖片若恢復原始檔名，就以該檔名來命名，且更新 mapping（影響下載時的檔名）
                if (original_title != "無") and (default_text != original_title):
                    new_text = original_title
                    new_filename = original_title + extension
                    removed_image_count += 1
                    label_ = st.session_state['image_labels'][selected_folder].get(outer_image_file, "無")
                    if label_ == "模特":
                        removed_model_count += 1
                    elif label_ == "平拍":
                        removed_flat_count += 1
                else:
                    if use_full_filename:
                        new_text = default_text
                        new_filename = default_text + extension
                    else:
                        new_text = default_text
                        new_filename = default_text + extension  # 外層圖片若沒有改動，也以預設 recovered 名稱命名
                        removed_image_count += 1
            else:
                if use_full_filename:
                    if new_text.strip() != default_text:
                        original_title = st.session_state['image_original_title'][selected_folder].get(outer_image_file, "無")
                        if default_text == original_title:
                            outer_moved_count += 1
                            added_image_count += 1
                            label_ = st.session_state['image_labels'][selected_folder].get(outer_image_file, "無")
                            if label_ == "模特":
                                added_model_count += 1
                            elif label_ == "平拍":
                                added_flat_count += 1
                    new_filename = new_text + extension
                else:
                    new_filename = prefix + new_text + extension
    
            if new_text.strip() != default_text:
                temp_filename_changes[outer_image_file] = {'new_filename': new_filename, 'text': new_text}
                if new_filename != '':
                    modified_outer_count += 1
    
        new_filenames = [
            data['new_filename'] for data in temp_filename_changes.values() if data['new_filename'] != ''
        ]
        duplicates = [filename for filename, count in Counter(new_filenames).items() if count > 1]
    
        if duplicates:
            st.session_state['has_duplicates'] = True
            st.session_state['duplicate_filenames'] = duplicates
            st.session_state['confirmed_changes'][selected_folder] = False
        else:
            st.session_state['has_duplicates'] = False
            st.session_state['confirmed_changes'][selected_folder] = True
    
            if not use_full_filename:
                def sort_key(item):
                    text_ = item[1]['text'].strip()
                    return (0, int(text_)) if text_.isdigit() else (1, text_)
        
                sorted_files = sorted(
                    ((file, data) for file, data in temp_filename_changes.items() if data['new_filename'] != ''),
                    key=sort_key
                )
                rename_counter = 1
                for file, data in sorted_files:
                    new_index = str(rename_counter).zfill(2)
                    ext_ = os.path.splitext(file)[1]
                    new_filename = f"{prefix}{new_index}{ext_}"
                    temp_filename_changes[file]['new_filename'] = new_filename
                    temp_filename_changes[file]['text'] = new_index
                    rename_counter += 1
        
            if selected_folder not in st.session_state['filename_changes']:
                st.session_state['filename_changes'][selected_folder] = {}
            st.session_state['filename_changes'][selected_folder].update(temp_filename_changes)
        
            for file, data in temp_filename_changes.items():
                text_input_key = f"{selected_folder}_{file}"
                st.session_state[text_input_key] = data['text']
        
            num_images_key = f"{selected_folder}_num_images"
            ad_images_key = f"{selected_folder}_ad_images"
            model_images_key = f"{selected_folder}_model_images"
            flat_images_key = f"{selected_folder}_flat_images"
        
            if not use_full_filename:
                total_count = 0
                model_count = 0
                flat_count = 0
                for file, data in st.session_state['filename_changes'][selected_folder].items():
                    if data['new_filename'].strip() != '':
                        total_count += 1
                        label_ = st.session_state['image_labels'][selected_folder].get(file, "無")
                        if label_ == "模特":
                            model_count += 1
                        elif label_ == "平拍":
                            flat_count += 1
                st.session_state[num_images_key] = str(total_count)
                st.session_state[model_images_key] = str(model_count)
                st.session_state[flat_images_key] = str(flat_count)
            else:
                if num_images_key in st.session_state:
                    current_num_images = int(st.session_state[num_images_key])
                    st.session_state[num_images_key] = str(
                        max(1, current_num_images - removed_image_count + modified_outer_count + added_image_count - outer_moved_count)
                    )
                current_model = int(st.session_state.get(model_images_key, 0))
                current_flat = int(st.session_state.get(flat_images_key, 0))
                new_model = current_model - removed_model_count + added_model_count
                new_flat = current_flat - removed_flat_count + added_flat_count
                st.session_state[model_images_key] = str(new_model)
                st.session_state[flat_images_key] = str(new_flat)
        
            ad_images_value = st.session_state.get(ad_images_key)
            data_ = folder_to_data.get(selected_folder, {})
            data_folder_name = data_.get('資料夾', selected_folder)
            st.session_state['folder_values'][data_folder_name] = {
                '張數': st.session_state[num_images_key],
                '廣告圖': ad_images_value,
                '模特': st.session_state[model_images_key],
                '平拍': st.session_state[flat_images_key],
            }
            st.session_state['modified_folders'].add(data_folder_name)
            
    def get_compress_type(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.psd', '.ai']:
            return zipfile.ZIP_STORED
        else:
            return zipfile.ZIP_DEFLATED

    def write_to_zip(zipf, file_path, arcname):
        ct = get_compress_type(file_path)
        if ct == zipfile.ZIP_DEFLATED:
            zipf.write(file_path, arcname, compress_type=ct, compresslevel=1)
        else:
            zipf.write(file_path, arcname, compress_type=ct)

    def clean_outer_images(zip_buffer):
        IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".psd", ".ai"]
        temp_dir = tempfile.mkdtemp()
        cleaned_zip_buffer = BytesIO()
        try:
            with zipfile.ZipFile(zip_buffer, "r") as zip_file:
                zip_file.extractall(temp_dir)
            found_2img = any("2-IMG" in dirs for _, dirs, _ in os.walk(temp_dir))
            if not found_2img:
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

    def cover_path_and_reset_key_tab2():
        if cover_path_input.strip():
            tmp_dir_path = st.session_state.get("custom_tmpdir")
            if tmp_dir_path:
                tmp_others_path = os.path.join(tmp_dir_path, "tmp_others")
                if os.path.exists(tmp_others_path):
                    shutil.rmtree(tmp_others_path, ignore_errors=True)
            def delete_file(file_path, file, ext):
                try:
                    if ('編圖結果' in file.lower() and file.lower().endswith('.xlsx')):
                        os.remove(file_path)
                    elif ext not in [".xlsx", ".gsheet", ".ai"]:
                        try:
                            os.remove(file_path)
                        except PermissionError:
                            try:
                                if os.name == 'nt':
                                    ctypes.windll.kernel32.SetFileAttributesW(file_path, 0x80)
                                    os.remove(file_path)
                                else:
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
                except PermissionError as e:
                    st.warning(f"無法刪除檔案: {file_path}，錯誤: {str(e)}")
            file_delete_tasks = []
            with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                for root, dirs, files in os.walk(cover_path_input, topdown=False):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        file_path = os.path.join(root, file)
                        if ('編圖結果' in file.lower() and file.lower().endswith('.xlsx')) or ext not in [".xlsx", ".gsheet", ".ai"]:
                            file_delete_tasks.append(executor.submit(delete_file, file_path, file, ext))
                for future in as_completed(file_delete_tasks):
                    future.result()
            dir_delete_tasks = []
            with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                for root, dirs, files in os.walk(cover_path_input, topdown=False):
                    for d in dirs:
                        if d == "0-上架資料":
                            continue
                        dir_path = os.path.join(root, d)
                        dir_delete_tasks.append(executor.submit(shutil.rmtree, dir_path, ignore_errors=True))
                for future in as_completed(dir_delete_tasks):
                    future.result()
            if "final_zip_content" in st.session_state and st.session_state["final_zip_content"]:
                final_zip_bytes = st.session_state["final_zip_content"]
                tmp_extract_dir = os.path.join(tempfile.gettempdir(), "cover_tmp")
                if os.path.exists(tmp_extract_dir):
                    shutil.rmtree(tmp_extract_dir, ignore_errors=True)
                os.makedirs(tmp_extract_dir, exist_ok=True)
                with zipfile.ZipFile(BytesIO(final_zip_bytes), 'r') as final_zip:
                    final_zip.extractall(tmp_extract_dir)
                tmp_others_extracted = os.path.join(tmp_extract_dir, "tmp_others")
                if os.path.exists(tmp_others_extracted):
                    shutil.rmtree(tmp_others_extracted, ignore_errors=True)
                def fast_copy_filtered(src, dst):
                    file_list = []
                    for root, dirs, files in os.walk(src):
                        rel_path = os.path.relpath(root, src)
                        target_dir = os.path.join(dst, rel_path)
                        os.makedirs(target_dir, exist_ok=True)
                        for file in files:
                            ext = os.path.splitext(file)[1].lower()
                            if ('編圖結果' in file.lower() and file.lower().endswith('.xlsx')) or ext not in [".xlsx", ".gsheet"]:
                                src_file = os.path.join(root, file)
                                dst_file = os.path.join(target_dir, file)
                                file_list.append((src_file, dst_file))
                    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                        futures = [executor.submit(shutil.copy2, src_file, dst_file) for src_file, dst_file in file_list]
                        for future in as_completed(futures):
                            future.result()
                fast_copy_filtered(tmp_extract_dir, cover_path_input)
                shutil.rmtree(tmp_extract_dir, ignore_errors=True)
            if "tmp_dir" in st.session_state and os.path.exists(st.session_state["tmp_dir"]):
                shutil.rmtree(st.session_state["tmp_dir"], ignore_errors=True)
                
        clear_all_caches()
        st.session_state['file_uploader_key2'] += 1
        st.session_state['text_area_key2'] += 1
        st.session_state['file_uploader_disabled_2'] = False
        st.session_state['text_area_disabled_2'] = False
        st.session_state['filename_changes'].clear()

    fixed_cache_base_dir = os.path.join(tempfile.gettempdir(), "streamlit_cache")
    fixed_psd_cache_dir = os.path.join(fixed_cache_base_dir, "psd_cache")
    fixed_ai_cache_dir = os.path.join(fixed_cache_base_dir, "ai_cache")
    fixed_custom_tmpdir = os.path.join(fixed_cache_base_dir, "custom_tmpdir")
    
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
        if not st.session_state.get("source_loaded", False):
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
                    copytree_multithreaded(input_path_2, tmpdirname)
            st.session_state["source_loaded"] = True

        excel_file_path = None
        found_excel_name = None
        for f in os.listdir(tmpdirname):
            if f.lower().endswith('.xlsx') and '編圖結果' in f:
                excel_file_path = os.path.join(tmpdirname, f)
                found_excel_name = f
                break
        if excel_file_path and os.path.exists(excel_file_path):
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
                    if data_folder_name in type_sheet_df['資料夾'].values:
                        type_idx = type_sheet_df[type_sheet_df['資料夾'] == data_folder_name].index[0]
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
                    '張數': '0',
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
            use_full_filename = True if os.path.exists(img_folder_path) else False
            if not use_full_filename:
                img_folder_path = os.path.join(tmpdirname, selected_folder, '1-Main', 'All')
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
                                image_path = (os.path.join(img_folder_path, image_file)
                                              if image_file in image_files
                                              else os.path.join(outer_folder_path, image_file))
                                if image_path not in st.session_state['image_cache'][selected_folder]:
                                    add_label = image_file.lower().endswith(('.png', '.tif', '.tiff', '.psd', '.ai'))
                                    image = load_and_process_image(image_path, add_label)
                                    st.session_state['image_cache'][selected_folder][image_path] = image
                                else:
                                    image = st.session_state['image_cache'][selected_folder][image_path]
                                col.image(image, use_container_width=True)
                                filename_without_ext = os.path.splitext(image_file)[0]
                                default_text = filename_without_ext if use_full_filename else (filename_without_ext.split('_', 1)[-1] if '_' in filename_without_ext else filename_without_ext)
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
                                if (data_folder_name and 'folder_values' in st.session_state and data_folder_name in st.session_state['folder_values']):
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
                                upper_limit = max(20, int(num_images_default), int(ad_images_default))
                                num_images_options = [str(i) for i in range(1, upper_limit + 1)]
                                ad_images_options = [str(i) for i in range(1, upper_limit + 1)]
                                type_images_options = [str(i) for i in range(0, 21)]
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
                        colA, colB, colC, colD = st.columns([3, 5, 8, 2.5], vertical_alignment="center")
                        if colA.form_submit_button(
                            "暫存修改",
                            on_click=handle_submission,
                            args=(selected_folder, images_to_display, outer_images_to_display, use_full_filename, folder_to_data)
                        ):
                            if st.session_state.get('has_duplicates') is False:
                                st.toast(f"資料夾 {selected_folder} 暫存修改成功!", icon='🎉')
                        if outer_images_to_display and not use_full_filename:
                            with colD.popover("外層圖片"):
                                outer_cols = st.columns(6)
                                for idx, outer_image_file in enumerate(outer_images_to_display):
                                    if idx % 6 == 0 and idx != 0:
                                        outer_cols = st.columns(6)
                                    col = outer_cols[idx % 6]
                                    outer_image_path = (os.path.join(outer_folder_path, outer_image_file)
                                                        if outer_image_file in outer_images
                                                        else os.path.join(img_folder_path, outer_image_file))
                                    if outer_image_path not in st.session_state['image_cache'][selected_folder]:
                                        add_label = outer_image_file.lower().endswith(('.png', '.tif', '.tiff', '.psd', '.ai'))
                                        outer_image = load_and_process_image(outer_image_path, add_label)
                                        st.session_state['image_cache'][selected_folder][outer_image_path] = outer_image
                                    else:
                                        outer_image = st.session_state['image_cache'][selected_folder][outer_image_path]
                                    col.image(outer_image, use_container_width=True)
                                    filename_without_ext = os.path.splitext(outer_image_file)[0]
                                    default_text = filename_without_ext if use_full_filename else (filename_without_ext.split('_', 1)[-1] if '_' in filename_without_ext else filename_without_ext)
                                    if (selected_folder in st.session_state['filename_changes']
                                            and outer_image_file in st.session_state['filename_changes'][selected_folder]):
                                        modified_text = st.session_state['filename_changes'][selected_folder][outer_image_file]['text']
                                        if modified_text == '':
                                            modified_text = st.session_state['filename_changes'][selected_folder][outer_image_file].get('last_non_empty', default_text)
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
                            st.write(st.session_state['filename_changes'][selected_folder])
                            def parallel_read_zip_compress():
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
                                                    # 若該檔案位於資料夾根目錄（外層圖片），path_parts 長度應為 2
                                                    if len(path_parts) == 2:
                                                        new_rel_path = os.path.join(folder_name, new_filename)
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
                                    # 依照原有邏輯，處理 Excel 檔案的合併……
                                    excel_buffer = BytesIO()
                                    if "編圖紀錄" in excel_sheets:
                                        df_result = excel_sheets["編圖紀錄"]
                                        if "編號" in df_result.columns:
                                            def fix_code(x):
                                                if pd.isna(x):
                                                    return ""
                                                s = str(x)
                                                if s.endswith(".0"):
                                                    s = s[:-2]
                                                return s.zfill(2)
                                            df_result["編號"] = df_result["編號"].apply(fix_code)
                                        excel_sheets["編圖紀錄"] = df_result
                        
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
                                                new_row = pd.DataFrame([{'資料夾': data_folder_name, '張數': num_images, '廣告圖': ad_images}])
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
                                                new_row = pd.DataFrame([{'資料夾': data_folder_name, '模特': model_images, '平拍': flat_images}])
                                                type_result_df = pd.concat([type_result_df, new_row], ignore_index=True)
                                        excel_sheets['圖片類型統計'] = type_result_df
                                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                            for sheet_name, df in excel_sheets.items():
                                                df.to_excel(writer, index=False, sheet_name=sheet_name)
                                                workbook = writer.book
                                                worksheet = writer.sheets[sheet_name]
                                                if sheet_name == '編圖紀錄' and '編號' in df.columns:
                                                    col_idx = df.columns.get_loc('編號')
                                                    text_format = workbook.add_format({'num_format': '@'})
                                                    worksheet.set_column(col_idx, col_idx, None, text_format)
                                                if sheet_name == '編圖張數與廣告圖':
                                                    text_format = workbook.add_format({'num_format': '@'})
                                                    worksheet.set_column(1, 2, None, text_format)
                                                elif sheet_name == '圖片類型統計':
                                                    text_format = workbook.add_format({'num_format': '@'})
                                                    worksheet.set_column(1, 2, None, text_format)
                                    else:
                                        result_df = pd.DataFrame(columns=['資料夾', '張數', '廣告圖'])
                                        type_result_df = pd.DataFrame(columns=['資料夾', '模特', '平拍'])
                                        for data_folder_name in st.session_state['folder_values']:
                                            num_images = st.session_state['folder_values'][data_folder_name]['張數']
                                            ad_images = st.session_state['folder_values'][data_folder_name]['廣告圖']
                                            ad_images = f"{int(ad_images):02}"
                                            new_row = pd.DataFrame([{'資料夾': data_folder_name, '張數': num_images, '廣告圖': ad_images}])
                                            result_df = pd.concat([result_df, new_row], ignore_index=True)
                                            model_images = st.session_state['folder_values'][data_folder_name]['模特']
                                            flat_images = st.session_state['folder_values'][data_folder_name]['平拍']
                                            new_type_row = pd.DataFrame([{'資料夾': data_folder_name, '模特': model_images, '平拍': flat_images}])
                                            type_result_df = pd.concat([type_result_df, new_type_row], ignore_index=True)
                                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                            result_df.to_excel(writer, index=False, sheet_name='編圖張數與廣告圖')
                                            type_result_df.to_excel(writer, index=False, sheet_name='圖片類型統計')
                                            workbook = writer.book
                                            worksheet1 = writer.sheets['編圖張數與廣告圖']
                                            worksheet2 = writer.sheets['圖片類型統計']
                                            text_format = workbook.add_format({'num_format': '@'})
                                            worksheet1.set_column(1, 2, None, text_format)
                                            worksheet2.set_column(1, 2, None, text_format)
                                    excel_buffer.seek(0)
                                    zipf.writestr(found_excel_name if found_excel_name else '編圖結果.xlsx', excel_buffer.getvalue())
                        
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
                            col1_, col2_, col3_, col4_ = st.columns([1.25,3.35,0.1,0.95],vertical_alignment="center")
                            if not uploaded_file_2 and input_path_2:
                                cover_text_default = input_path_2.strip()
                            elif st.session_state.get("input_path_from_tab1"):
                                cover_text_default = st.session_state.get("input_path_from_tab1")
                            else:
                                cover_text_default = ""
                            global cover_path_input
                            
                            col1_.write("同步覆蓋此路徑的檔案(選填)")
                            cover_path_input = col2_.text_input(
                                label="同步覆蓋此路徑的檔案",
                                value=cover_text_default, 
                            )
                            col4_.download_button(
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
