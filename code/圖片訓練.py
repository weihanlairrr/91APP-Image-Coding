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
import pymupdf
import torch
import pickle
import urllib.request 
from io import BytesIO
from psd_tools import PSDImage
from PIL import Image, ImageOps, ImageDraw, ImageFont, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from streamlit_extras.stylable_container import stylable_container

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
            if os.path.exists(cache_base_dir):
                shutil.rmtree(cache_base_dir, ignore_errors=False, onerror=on_rm_error)
            os.makedirs(cache_base_dir, exist_ok=True)
            os.makedirs(psd_cache_dir, exist_ok=True)
            os.makedirs(ai_cache_dir, exist_ok=True)
            os.makedirs(custom_tmp_dir, exist_ok=True)
        if mode in ("reinitialize", "both"):
            keys = [
                'image_cache', 'modified_folders',
                'previous_input_path', 'file_uploader_disabled_3', 'text_area_disabled_3',
                'custom_tmpdir', 'previous_selected_folder', 'source_loaded',
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
            'custom_tmpdir': custom_tmp_dir,
            'previous_selected_folder': None,
            'source_loaded': False,
        }
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)

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
        st.session_state["custom_tmpdir"] = custom_tmp_dir
        file_key = 'file_uploader_' + str(st.session_state.get('file_uploader_key3', 0))
        uploaded_file_1 = st.session_state.get(file_key, None)
        if uploaded_file_1:
            st.session_state['image_cache'].clear()
            st.session_state["source_loaded"] = False
        st.session_state.text_area_disabled_3 = bool(uploaded_file_1)

    def handle_text_area_change_tab3():
        initialize_tab3(mode="both")
        st.session_state["custom_tmpdir"] = custom_tmp_dir
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
        psd_cache = psd_cache_dir
        ai_cache = ai_cache_dir
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
                    doc = pymupdf.open(image_path)
                    page = doc.load_page(0)
                    pix = page.get_pixmap(dpi=100)
                    image = Image.open(BytesIO(pix.tobytes("png")))
                    doc.close()
                    image.save(cache_path, format='PNG')
                except Exception as e:
                    raise Exception(f"無法處理 .ai 檔案: {str(e)}")
        else:
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

    def handle_submit_angles(folder, angle_widget_keys):
        """
        將使用者的角度選擇寫入 session_state，
        並清除該資料夾所有選項的 widget 狀態。
        """
        angle_selections = {}
        for key, img_rel in angle_widget_keys:
            angle_val = st.session_state.get(key, "")
            angle_selections[img_rel] = angle_val

        if "angle_changes" not in st.session_state:
            st.session_state["angle_changes"] = {}
        st.session_state["angle_changes"][folder] = angle_selections

        for key, _ in angle_widget_keys:
            if key in st.session_state:
                del st.session_state[key]

        st.toast(f"資料夾 {folder} 暫存修改成功!", icon='🎉')

    def get_image_features(image, model):
        """
        使用 ResNet 模型與預處理，萃取圖片特徵。
        """
        image_tensor = image_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model(image_tensor).detach().cpu().numpy().flatten()
        return feats

    def load_existing_features(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def remove_duplicates(features_map):
        """
        移除 features_map 中重複的 labeled_features，
        並回傳更新後的資料與刪除筆數。
        """
        duplicate_count = 0
        for b_, categories_ in features_map.items():
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
                        duplicate_count += 1
                features_map[b_][cat_key]["labeled_features"] = unique_labeled_features
        return features_map, duplicate_count

    def reset_tab3():
        st.session_state['file_uploader_key3'] += 1
        st.session_state['text_area_key3'] += 1
        st.session_state['file_uploader_disabled_3'] = False
        st.session_state['text_area_disabled_3'] = False
        initialize_tab3()

    # =============================================================================
    # 全域變數 - 暫存資料夾路徑設定
    # =============================================================================
    cache_base_dir = os.path.join(tempfile.gettempdir(), "streamlit_cache")
    psd_cache_dir = os.path.join(cache_base_dir, "psd_cache")
    ai_cache_dir = os.path.join(cache_base_dir, "ai_cache")
    custom_tmp_dir = os.path.join(cache_base_dir, "custom_tmpdir")

    # =============================================================================
    # 介面初始化與檔案上傳/資料夾路徑輸入
    # =============================================================================
    initialize_tab3()
    st.write("\n")
    col1, col2 = st.columns(2, vertical_alignment="top")
    with col1:
        with stylable_container(
            key="file_uploader",
            css_styles="""
            {
              [data-testid='stFileUploaderDropzoneInstructions'] > div > span {
                display: none;
              }
              [data-testid='stFileUploaderDropzoneInstructions'] > div::before {
                content: '上傳 ZIP 或 EXCEL';
              }
            }
            """,
            ):
            uploaded_file_3 = st.file_uploader(
                "上傳編圖結果 ZIP 或 XLSX 檔",
                type=["zip", "xlsx"],
                key='file_uploader_' + str(st.session_state['file_uploader_key3']),
                disabled=st.session_state['file_uploader_disabled_3'],
                on_change=handle_file_uploader_change_tab3,
                label_visibility="collapsed"
            )
    with col2: 
        input_path_3 = st.text_area(
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

        # 1) 處理 user 上傳的 .xlsx（跳過原本的 form 與 popover）
        if uploaded_file_3 and uploaded_file_3.name.lower().endswith(".xlsx"):
            # 讀取使用者已完成的結果表
            try:
                df_upload = pd.read_excel(uploaded_file_3)
            except Exception as e:
                st.error(f"讀取上傳的 Excel 時發生錯誤：{e}")
                st.stop()
            required_cols = ["URL", "品牌", "商品分類", "角度", "編號"]
            if list(df_upload.columns) != required_cols:
                st.error("上傳的 Excel 欄位與格式不符，請確認欄位為 URL、品牌、商品分類、角度、編號")
                st.stop()
            df_complete = df_upload.dropna(subset=required_cols)
            df_complete = df_complete[
                df_complete[required_cols]
                .applymap(lambda x: str(x).strip() != "")
                .all(axis=1)
            ]
            if df_complete.empty:
                st.error("您所上傳的檔案缺少資料，請補正後重新上傳")
                st.stop()
            # 顯示上傳內容
            st.dataframe(df_upload, use_container_width=True, hide_index=True,height=457)
            # 開始執行按鈕
            button_placeholder = st.empty()
            with button_placeholder:
                start_running = st.button('開始執行', key='start_running_from_xlsx')
            if start_running:
                button_placeholder.empty()
                st.write("\n")
                df_final = df_upload
                # 品牌取第一筆資料
                brand = df_upload["品牌"].dropna().iloc[0]

                # ---------------------
                # 進入 ResNet 特徵萃取流程（與原本相同）
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

                image_preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
                dep_dir = os.path.join(
                    r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies",
                    brand
                )
                # 偵測 mapping Excel 檔案
                mapping_files = [
                    f for f in os.listdir(dep_dir)
                    if f.lower().endswith('.xlsx') and '檔名角度對照表' in f
                ]
                pkl_path = os.path.join(dep_dir, f"{brand}_image_features.pkl")
                if mapping_files and not os.path.exists(pkl_path):
                    mapping_df = pd.read_excel(os.path.join(dep_dir, mapping_files[0]))
                    created = {brand: {}}
                    for _, row in mapping_df.iterrows():
                        fname = str(row["對應檔名"]).strip()
                        cat   = str(row["檔名專屬的商品分類"]).strip()
                        ang   = str(row["角度"]).strip()
                        img_path = os.path.join(tmpdirname, fname)
                        try:
                            img = Image.open(img_path).convert("RGB")
                        except Exception as e:
                            st.warning(f"建立 .pkl 時無法開啟圖片：{fname}，錯誤：{e}")
                            continue
                        feat = get_image_features(img, resnet)
                        if cat not in created[brand]:
                            created[brand][cat] = {"all_features": [], "labeled_features": []}
                        num = len(created[brand][cat]["labeled_features"]) + 1
                        created[brand][cat]["all_features"].append(feat)
                        created[brand][cat]["labeled_features"].append({
                            "features": feat,
                            "labels": {"angle": ang, "number": num}
                        })
                    with open(pkl_path, "wb") as f:
                        pickle.dump(created, f)
                    
                progress_bar = st.progress(0)
                progress_text = st.empty()
                total = len(df_final)
                processed_images_count = 0

                for brand_name in df_final["品牌"].unique():
                    brand_df = df_final[df_final["品牌"] == brand_name]
                    output_dir = os.path.join(
                        r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies",
                        brand_name
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    pkl_file_path = os.path.join(output_dir, f"{brand_name}_image_features.pkl")
                    features_by_category = load_existing_features(pkl_file_path)

                    # 計算舊有資料筆數
                    original_len = 0
                    if brand_name in features_by_category:
                        for cat_key, cat_data in features_by_category[brand_name].items():
                            original_len += len(cat_data["labeled_features"])

                    for idx, row_ in brand_df.iterrows():
                        url_ = row_["URL"]
                        category_ = row_["商品分類"]
                        angle_ = row_["角度"]
                        number_ = row_["編號"]

                        # 下載並開啟遠端圖片
                        try:
                            with urllib.request.urlopen(url_) as resp:
                                data = resp.read()
                            img = Image.open(BytesIO(data)).convert('RGB')
                        except Exception as e:
                            st.warning(f"下載或開啟圖片失敗：{url_}，錯誤：{e}")
                            processed_images_count += 1
                            progress_bar.progress(int(processed_images_count / total * 100))
                            progress_text.write(f"正在處理 {processed_images_count} / {total}")
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

                        processed_images_count += 1
                        progress_bar.progress(int(processed_images_count / total * 100))
                        progress_text.write(f"正在處理 {processed_images_count} / {total}")

                    # 去重並寫回 pkl
                    progress_bar.empty()
                    progress_text.empty()

                    with st.spinner(" 尚未結束，請勿關閉"):
                        features_by_category, dupe_count = remove_duplicates(features_by_category)
                        with open(pkl_file_path, 'wb') as f:
                            pickle.dump(features_by_category, f)

                        # 重新計算去重後的筆數
                        new_len = 0
                        if brand_name in features_by_category:
                            for cat_key, cat_data in features_by_category[brand_name].items():
                                new_len += len(cat_data["labeled_features"])

                        added_count = new_len - original_len

                        if dupe_count > 0:
                            st.success(f"{brand_name}圖片資料訓練完成！共新增{added_count}筆資料 (發現{dupe_count}筆重複資料)，總資料數為{new_len}筆")
                        else:
                            st.success(f"{brand_name}圖片資料訓練完成！共新增{added_count}筆資料，總資料數為{new_len}筆")

                if st.button("結束", on_click=reset_tab3()):
                    st.rerun()

            else:
                st.stop()
            return  # 完成 .xlsx 分支後結束

        # 2) 原本的 ZIP / 資料夾路徑 分支
        if not st.session_state.get("source_loaded", False):
            if uploaded_file_3 and uploaded_file_3.name.lower().endswith(".zip"):
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
                st.error("此檔案不支援，請上傳複檢後下載的檔案")
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
            st.error(f"在品牌資料夾中未找到 .xlsx 檔案：{brand_dataset_path}")
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
            load_futures = []
            with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
                if os.path.exists(img_folder_path):
                    image_files = get_outer_folder_images(img_folder_path)
                    for image_file in image_files:
                        image_path = os.path.join(img_folder_path, image_file)
                        if image_path not in st.session_state['image_cache'][folder]:
                            add_label = image_file.lower().endswith(('.png', '.tif', '.tiff', '.psd', '.ai'))
                            future = executor.submit(load_and_process_image, image_path, add_label)
                            load_futures.append((future, folder, image_path))
                outer_folder_path = os.path.join(tmpdirname, folder)
                outer_images = get_outer_folder_images(outer_folder_path)
                for outer_image_file in outer_images:
                    image_path = os.path.join(outer_folder_path, outer_image_file)
                    if image_path not in st.session_state['image_cache'][folder]:
                        add_label = outer_image_file.lower().endswith(('.png', '.tif', '.tiff', '.psd', '.ai'))
                        future = executor.submit(load_and_process_image, image_path, add_label)
                        load_futures.append((future, folder, image_path))
                for future, folder_, image_path_ in load_futures:
                    try:
                        image_ = future.result()
                        st.session_state['image_cache'][folder_][image_path_] = image_
                    except Exception as e:
                        st.warning(f"載入圖片 {image_path_} 時發生錯誤: {str(e)}")

        # =============================================================================
        # 顯示資料夾與角度選擇介面（含開啟 Excel 與同步更新 .pkl 編號）
        # =============================================================================
        if top_level_folders:
            if 'previous_selected_folder' not in st.session_state and top_level_folders:
                st.session_state['previous_selected_folder'] = top_level_folders[0]
            if top_level_folders:
                if 'previous_selected_folder' not in st.session_state:
                    st.session_state['previous_selected_folder'] = None
                col1_, col2_,= st.columns([7.5, 1], vertical_alignment="bottom")
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
                

                # 以下為原有的角度選擇 form
                df_category = target_sheet[target_sheet["商品分類"] == category]
                angle_options_all = df_category["角度"].dropna().unique().tolist()
                angle_options_all = [str(a).strip() for a in angle_options_all if str(a).strip() != ""]
                angle_options_all = [a for a in angle_options_all if re.search(r'[\u4e00-\u9fff]', a)]
                saved_angles = st.session_state.get("angle_changes", {}).get(selected_folder, {})
                angle_widget_keys = []
                
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
                        # 如果品牌為 ADS 且讀取的角度為 D1～D5，則等同於細節
                        if brand == "ADS" and default_angle in {"D1", "D2", "D3", "D4", "D5"}:
                            default_angle = "細節"
                        angle_options_with_skip = ["不訓練"] + angle_options_all
                        if default_angle == "不訓練":
                            default_index = 0
                        elif default_angle in angle_options_all:
                            default_index = angle_options_all.index(default_angle) + 1
                        else:
                            default_index = None
                        angle_key = f"{selected_folder}_angle_{image_relative_path}"
                        angle_widget_keys.append((angle_key, image_relative_path))
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
                        args=(selected_folder, angle_widget_keys)
                    )
                    with colD.popover("訓練資料預覽"):
                        preview_records = []
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
                            preview_records.append({
                                "URL": url_,
                                "品牌": brand_,
                                "商品分類": final_category,
                                "角度": final_angle,
                                "編號": final_number
                            })
                        if preview_records:
                            df_preview = pd.DataFrame(preview_records, columns=["URL", "品牌", "商品分類", "角度", "編號"])
                            st.dataframe(df_preview, use_container_width=True, hide_index=True)
                            # 將訓練用的 df_preview 存到 session_state 供後續使用
                            st.session_state["df_preview"] = df_preview
                        else:
                            st.warning("暫無資料。")
        else:
            st.error("未找到任何資料夾。")

        # =============================================================================
        # ResNet 訓練 (特徵萃取) 邏輯 - ZIP 分支
        # =============================================================================
        colA, colB = st.columns([7.5,1])
        with colB.popover("新增角度"):
            col5,col6 = st.columns([3,1],vertical_alignment="center")
            col5.info("步驟 1: 開啟角度清單 > 在 **工作表2** 新增角度並調整受影響的編號")
            if col6.button("開啟角度清單", key=f"open_excel_{brand}"):
                if dataset_xlsx and os.path.exists(dataset_xlsx):
                    try:
                        os.startfile(dataset_xlsx)
                    except Exception as e:
                        st.error(f"無法開啟檔案：{e}")
                else:
                    st.warning(f"找不到檔案：{dataset_xlsx}")
            
            col6,col7 = st.columns([3,1],vertical_alignment="center")
            col6.info("步驟 2: EXCEL修改後儲存關閉 > 點擊 **更新模型資料**")
            if col7.button("更新模型資料", key=f"sync_pkl_{brand}"):
                # 設定 pkl 路徑
                pkl_dir = os.path.join(
                    r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies",
                    brand
                )
                pkl_file = os.path.join(pkl_dir, f"{brand}_image_features.pkl")
                if not os.path.exists(pkl_file):
                    st.error(f"找不到 pkl 檔案：{pkl_file}")
                else:
                    with st.spinner("模型資料更新中"):
                        features_map = load_existing_features(pkl_file)
                        updated = {}
                        # 只處理這個品牌的資料
                        if brand in features_map:
                            for cat_key, cat_data in features_map[brand].items():
                                for item in cat_data["labeled_features"]:
                                    old_num = item["labels"]["number"]
                                    angle = item["labels"]["angle"]
                                    category_ = item["labels"]["category"]
                                    new_num = angle_map.get((category_, angle), old_num)
                                    if str(new_num) != str(old_num):
                                        # 登記一次
                                        key = (category_, angle)
                                        if key not in updated:
                                            updated[key] = (old_num, new_num)
                                        # 更新所有項目的 number
                                        item["labels"]["number"] = new_num
                        # 寫回 pkl
                        with open(pkl_file, 'wb') as f:
                            pickle.dump(features_map, f)
                        if updated:
                            # 將更新內容轉為 DataFrame 顯示
                            df_updates = pd.DataFrame([{
                                "商品分類": cat,
                                "角度": ang,
                                "修改前編號": old,
                                "修改後編號": new
                            } for (cat, ang), (old, new) in updated.items()])
                            with st.container(height=300):
                                st.write("更新完成！以下為更新內容：")
                                st.dataframe(df_updates, use_container_width=True)
                        else:
                            st.info("模型資料已是最新版。")
        if colA.checkbox("所有資料夾均確認完成"):
            button_placeholder = st.empty()
            with button_placeholder:
                start_running = st.button('開始執行')
            if start_running:
                button_placeholder.empty()
                st.write("\n")
                # 改為直接使用上方 popover 產生的 df_preview
                if "df_preview" not in st.session_state or st.session_state["df_preview"].empty:
                    st.warning("沒有任何可訓練的資料.")
                    st.stop()

                df_final = st.session_state["df_preview"]

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

                image_preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
                
                dep_dir = os.path.join(
                    r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies",
                    brand
                )
                # 偵測 mapping Excel 檔案
                mapping_files = [
                    f for f in os.listdir(dep_dir)
                    if f.lower().endswith('.xlsx') and '檔名角度對照表' in f
                ]
                pkl_path = os.path.join(dep_dir, f"{brand}_image_features.pkl")
                if mapping_files and not os.path.exists(pkl_path):
                    mapping_df = pd.read_excel(os.path.join(dep_dir, mapping_files[0]))
                    created = {brand: {}}
                    for _, row in mapping_df.iterrows():
                        fname = str(row["對應檔名"]).strip()
                        cat   = str(row["檔名專屬的商品分類"]).strip()
                        ang   = str(row["角度"]).strip()
                        img_path = os.path.join(tmpdirname, fname)
                        try:
                            img = Image.open(img_path).convert("RGB")
                        except Exception as e:
                            st.warning(f"建立 .pkl 時無法開啟圖片：{fname}，錯誤：{e}")
                            continue
                        feat = get_image_features(img, resnet)
                        if cat not in created[brand]:
                            created[brand][cat] = {"all_features": [], "labeled_features": []}
                        num = len(created[brand][cat]["labeled_features"]) + 1
                        created[brand][cat]["all_features"].append(feat)
                        created[brand][cat]["labeled_features"].append({
                            "features": feat,
                            "labels": {"angle": ang, "number": num}
                        })
                    with open(pkl_path, "wb") as f:
                        pickle.dump(created, f)
                    
                progress_bar = st.progress(0)
                progress_text = st.empty()
                total = len(df_final)
                processed_images_count = 0

                for brand_name in df_final["品牌"].unique():
                    brand_df = df_final[df_final["品牌"] == brand_name]

                    output_dir = os.path.join(
                        r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies",
                        brand_name
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    pkl_file_path = os.path.join(output_dir, f"{brand_name}_image_features.pkl")
                    features_by_category = load_existing_features(pkl_file_path)

                    original_len = 0
                    if brand_name in features_by_category:
                        for cat_key, cat_data in features_by_category[brand_name].items():
                            original_len += len(cat_data["labeled_features"])

                    for idx, row_ in brand_df.iterrows():
                        image_path = os.path.join(tmpdirname, row_["URL"])
                        category_ = row_["商品分類"]
                        angle_ = row_["角度"]
                        number_ = row_["編號"]

                        if not os.path.exists(image_path):
                            st.warning(f"找不到檔案：{image_path}")
                            processed_images_count += 1
                            progress_bar.progress(int(processed_images_count / total * 100))
                            progress_text.write(f"正在處理 {processed_images_count} / {total}")
                            continue

                        try:
                            img = Image.open(image_path).convert('RGB')
                        except UnidentifiedImageError:
                            st.warning(f"無法識別圖片：{image_path}")
                            processed_images_count += 1
                            progress_bar.progress(int(processed_images_count / total * 100))
                            progress_text.write(f"正在處理 {processed_images_count} / {total}")
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

                        processed_images_count += 1
                        progress_bar.progress(int(processed_images_count / total * 100))
                        progress_text.write(f"正在處理 {processed_images_count} / {total}")

                    progress_bar.empty()
                    progress_text.empty()

                    with st.spinner(" 尚未結束，請勿關閉"):
                        features_by_category, dupe_count = remove_duplicates(features_by_category)
                        with open(pkl_file_path, 'wb') as f:
                            pickle.dump(features_by_category, f)

                        new_len = 0
                        if brand_name in features_by_category:
                            for cat_key, cat_data in features_by_category[brand_name].items():
                                new_len += len(cat_data["labeled_features"])

                        added_count = new_len - original_len

                        if dupe_count > 0:
                            st.success(f"{brand_name}圖片資料訓練完成！共新增{added_count}筆資料 (發現{dupe_count}筆重複資料)，總資料數為{new_len}筆")
                        else:
                            st.success(f"{brand_name}圖片資料訓練完成！共新增{added_count}筆資料，總資料數為{new_len}筆")

                if st.button("結束", on_click=reset_tab3()):
                    st.rerun()
