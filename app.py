import streamlit as st
import pandas as pd
import zipfile
import os
import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import pickle
import shutil
import numpy as np
import platform

custom_css = """
    <style>
    .main {
        padding-left: 29%; 
        padding-right: 29%;
    }
    div.block-container{padding-top:4rem;
    }
    .stButton > button {
        padding: 5px 30px;
        background: #5A5B5E!important;
        color: #f5f5f5!important;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 5px 0;
    }
    .stDownloadButton button {
        background: #5A5B5E!important;
        color: #f5f5f5!important;
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
    .stButton > button:hover  {
        background: #8A8B8D!important;
    }
    .stDownloadButton button:hover {
        background: #8A8B8D!important;
    }
    button:hover  {
        background: #D3D3D3!important;
    }
    </style>
    """

st.markdown(custom_css, unsafe_allow_html=True)

# 定義提取特徵的函數
def get_image_features(image, model):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()
    return features

# 定義餘弦相似度計算
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def reset_file_uploader():
    st.session_state['file_uploader_key1'] += 1
    st.session_state['file_uploader_key2'] += 1
    if os.path.exists("uploaded_images"):
        shutil.rmtree("uploaded_images")
    if os.path.exists("temp.zip"):
        os.remove("temp.zip") 

# 解壓 zip 檔案並處理圖片
def unzip_file(uploaded_zip):
    system = platform.system()
    
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        for member in zip_ref.infolist():
            if system == "Windows":
                try:
                    member.filename = member.filename.encode('utf-8').decode('utf-8')
                except UnicodeDecodeError:
                    member.filename = member.filename.encode('utf-8').decode('latin1')
            elif system == "Darwin":
                try:
                    member.filename = member.filename.encode('cp437').decode('utf-8')
                except UnicodeDecodeError:
                    member.filename = member.filename.encode('cp437').decode('latin1')
            else:
                try:
                    member.filename = member.filename.encode('utf-8').decode('utf-8')
                except UnicodeDecodeError:
                    member.filename = member.filename.encode('utf-8').decode('latin1')
            
            zip_ref.extract(member, "uploaded_images")

# 定義需要跳過的關鍵字
keywords_to_skip = ["_SL_","_SLB_", "_SMC_", "_FR_", "_Fout_", "-1", "_Sid_", "_SUPV_", "_SUPS_", "_HM_","_BL_","_FM_","_BSM_","_LSL_","Thumbs"]

def rename_numbers_in_folder(results):
    folders = set([result["資料夾名稱"] for result in results])
    for folder in folders:
        folder_results = [r for r in results if r["資料夾名稱"] == folder]
        if any(pd.isna(r["編號"]) or r["編號"] == "" for r in folder_results):
            continue
        folder_results.sort(key=lambda x: int(x["編號"]))
        for idx, result in enumerate(folder_results):
            if idx < 10:
                result["編號"] = f'{idx+1:02}'
            else:
                result["編號"] = "超過上限"

    return results

# 重命名並打包資料夾及 Excel 檔案
def rename_and_zip_folders(results, output_excel_data, skipped_images):
    for result in results:
        folder_name = result["資料夾名稱"]
        image_file = result["圖片"]
        new_number = result["編號"]
    
        folder_path = os.path.join("uploaded_images", folder_name)
        main_folder_path = os.path.join(folder_path, "1-Main")
        all_folder_path = os.path.join(main_folder_path, "All")
        os.makedirs(all_folder_path, exist_ok=True)
        
        old_image_path = os.path.join(folder_path, image_file)

        if new_number == "超過上限":
            new_image_path = os.path.join(folder_path, image_file)
        else:
            new_image_name = f"{folder_name}_{new_number}.jpg"
            new_image_path = os.path.join(all_folder_path, new_image_name)
    
        if os.path.exists(old_image_path):
            os.rename(old_image_path, new_image_path)

    for skipped_image in skipped_images:
        folder_name = skipped_image["資料夾名稱"]
        image_file = skipped_image["圖片"]
        folder_path = os.path.join("uploaded_images", folder_name)
        old_image_path = os.path.join(folder_path, image_file)
        
        if os.path.exists(old_image_path):
            new_image_path = os.path.join(folder_path, image_file)
            os.rename(old_image_path, new_image_path)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder in os.listdir("uploaded_images"):
            folder_path = os.path.join("uploaded_images", folder)
            if os.path.isdir(folder_path):
                new_folder_name = f"{folder}_OK"
                new_folder_path = os.path.join("uploaded_images", new_folder_name)
                os.rename(folder_path, new_folder_path)
                
                for root, dirs, files in os.walk(new_folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, "uploaded_images"))
        
        zipf.writestr("編圖結果.xlsx", output_excel_data)

    return zip_buffer.getvalue()

    
# 初始化 ResNet 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  
resnet.eval().to(device)

# 圖片預處理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加載保存的圖片特徵
with open('image_features.pkl', 'rb') as f:
    features_by_category = pickle.load(f)

if 'file_uploader_key1' not in st.session_state:
    st.session_state['file_uploader_key1'] = 0
if 'file_uploader_key2' not in st.session_state:
    st.session_state['file_uploader_key2'] = 1
    
st.header("TP 編圖工具")
st.write("\n")

uploaded_zip = st.file_uploader("上傳 zip 檔案", type=["zip"], key='file_uploader_' + str(st.session_state['file_uploader_key1']))

selectbox_placeholder = st.empty()
button_placeholder = st.empty()
if uploaded_zip:
    with selectbox_placeholder:
        selected_brand = st.selectbox("請選擇品牌", list(features_by_category.keys()))
    with button_placeholder:
        start_running = st.button("開始執行")
    
if uploaded_zip and start_running:
    selectbox_placeholder.empty()
    button_placeholder.empty()
    st.write("\n")
    
    if os.path.exists("uploaded_images"):
        shutil.rmtree("uploaded_images")
        
    with open("temp.zip", "wb") as f:
        f.write(uploaded_zip.getbuffer())

    unzip_file("temp.zip")

    special_angle_mappings = {}
    if selected_brand == "ADS":
        df_angles = pd.read_excel("ADS檔名角度對照表.xlsx")
        for idx, row in df_angles.iterrows():
            keyword = str(row.iloc[0]).strip()
            angle = str(row.iloc[1]).strip()
            special_angle_mappings[keyword] = angle

    image_folders = [f for f in os.listdir("uploaded_images") if os.path.isdir(os.path.join("uploaded_images", f)) and not f.startswith('__MACOSX')]
    results = []
    skipped_images = []
    progress_bar = st.progress(0)
    progress_text = st.empty()

    total_folders = len(image_folders)
    processed_folders = 0

    for folder in image_folders:
        folder_path = os.path.join("uploaded_images", folder)
        image_files = os.listdir(folder_path)
        folder_features = []

        progress_text.text(f"正在處理資料夾: {folder}")

        special_images = []

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            if image_file.startswith('.') or os.path.isdir(image_path):
                continue
            
            if any(keyword in image_file for keyword in keywords_to_skip):
                skipped_images.append({"資料夾名稱": folder, "圖片": image_file})
                continue

            special_angle = None
            if special_angle_mappings:
                for substr, angle in special_angle_mappings.items():
                    if substr in image_file:
                        special_angle = angle
                        break

            img = Image.open(image_path).convert('RGB')
            img_features = get_image_features(img, resnet)
            folder_features.append({
                "image_file": image_file,
                "features": img_features,
                "special_angle": special_angle
            })

            if special_angle:
                special_images.append({
                    "image_file": image_file,
                    "special_angle": special_angle
                })

        best_category = None
        best_similarity = -1

        if len(folder_features) == 0:
            st.warning(f"資料夾 {folder} 中沒有有效的圖片，跳過此資料夾")
            continue

        for img_data in folder_features:
            img_features = img_data["features"]

            for brand in features_by_category:
                for category in features_by_category[brand]:
                    for item in features_by_category[brand][category]["labeled_features"]:
                        item_features = item["features"]
                        similarity = cosine_similarity(img_features, item_features)

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_category = item["labels"]

        filtered_by_category = features_by_category[selected_brand][best_category["category"]]["labeled_features"]

        angle_to_number = {item["labels"]["angle"]: item["labels"]["number"] for item in filtered_by_category}

        used_angles = set()

        final_results = {}

        for img_data in folder_features:
            image_file = img_data["image_file"]
            special_angle = img_data["special_angle"]

            if special_angle:
                if special_angle != "細節" and special_angle in used_angles:
                    st.warning(f"資料夾 '{folder}' 中的圖片 '{image_file}' 角度 '{special_angle}' 已被使用，無法分配")
                    final_results[image_file] = None
                    continue
                else:
                    used_angles.add(special_angle)
                    if special_angle in angle_to_number:
                        label_info = {
                            "資料夾名稱": folder,
                            "圖片": image_file,
                            "品牌": selected_brand,
                            "商品分類": best_category["category"],
                            "角度": special_angle,
                            "編號": angle_to_number[special_angle],
                            "預測信心": "100%"
                        }
                        final_results[image_file] = label_info
                    else:
                        st.warning(f"商品分類 '{best_category['category']}' 中沒有角度 '{special_angle}'，圖片 '{image_file}' 無法分配")
                        final_results[image_file] = None
            else:
                final_results[image_file] = None

        non_special_images = [img_data for img_data in folder_features if not img_data["special_angle"]]

        if not special_angle_mappings:
            non_special_images = folder_features

        image_similarity_store = {}

        for img_data in non_special_images:
            image_file = img_data["image_file"]
            img_features = img_data["features"]
            image_similarity_list = []
            for item in filtered_by_category:
                item_features = item["features"]
                similarity = cosine_similarity(img_features, item_features)

                image_similarity_list.append({
                    "image_file": image_file,
                    "similarity": similarity,
                    "label": item["labels"],
                    "folder": folder
                })

            image_similarity_list.sort(key=lambda x: x["similarity"], reverse=True)
            unique_labels = []
            for candidate in image_similarity_list:
                if candidate["label"]["angle"] not in [label["label"]["angle"] for label in unique_labels]:
                    unique_labels.append(candidate)
                if len(unique_labels) == 10:
                    break

            image_similarity_store[image_file] = unique_labels

        for image_file, similarity_list in image_similarity_store.items():
            if final_results[image_file] is not None:
                continue

            first_label_angle = similarity_list[0]["label"]["angle"]
            if first_label_angle == "細節" or first_label_angle not in used_angles:
                used_angles.add(first_label_angle)
                final_results[image_file] = {
                    "資料夾名稱": similarity_list[0]["folder"],
                    "圖片": image_file,
                    "品牌": similarity_list[0]["label"]["brand"],
                    "商品分類": similarity_list[0]["label"]["category"],
                    "角度": similarity_list[0]["label"]["angle"],
                    "編號": similarity_list[0]["label"]["number"],
                    "預測信心": f"{similarity_list[0]['similarity'] * 100:.2f}%"
                }
            else:
                continue

        for image_file, similarity_list in image_similarity_store.items():
            if final_results[image_file] is not None:
                continue

            for i in range(1, len(similarity_list)):
                candidate_label = similarity_list[i]["label"]
                candidate_angle = candidate_label["angle"]
                if candidate_angle == "細節" or candidate_angle not in used_angles:
                    used_angles.add(candidate_angle)
                    final_results[image_file] = {
                        "資料夾名稱": similarity_list[i]["folder"],
                        "圖片": image_file,
                        "品牌": candidate_label["brand"],
                        "商品分類": candidate_label["category"],
                        "角度": candidate_angle,
                        "編號": candidate_label["number"],
                        "預測信心": f"{similarity_list[i]['similarity'] * 100:.2f}%"
                    }
                    break
                else:
                    continue

        for image_file, assignment in final_results.items():
            if assignment is not None:
                results.append(assignment)

        processed_folders += 1
        progress_bar.progress(processed_folders / total_folders)

    progress_bar.empty()
    progress_text.empty()

    results = rename_numbers_in_folder(results)

    result_df = pd.DataFrame(results)
    st.dataframe(result_df, hide_index=True, use_container_width=True)

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, index=False)
    excel_data = excel_buffer.getvalue()

    zip_data = rename_and_zip_folders(results, excel_data, skipped_images)

    if st.download_button(
        label="下載編圖結果",
        data=zip_data,
        file_name="編圖結果.zip",
        mime="application/zip",
        on_click=reset_file_uploader
    ):
        st.rerun()
