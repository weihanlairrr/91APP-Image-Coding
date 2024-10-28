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
from torchvision.models import ResNet50_Weights

st.set_page_config(page_title='TP自動化編圖工具', page_icon='👕')

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

# 初始化 ResNet 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  
resnet.eval().to(device)

# 圖片預處理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

# 定義需要跳過的關鍵字
keywords_to_skip = ["_SL_","_SLB_", "_SMC_", "_Fout_", "-1", "_Sid_", "_BL_","_FM_","_BSM_","_LSL_","Thumbs"]

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
    if os.path.exists("uploaded_images"):
        shutil.rmtree("uploaded_images")
    if os.path.exists("temp.zip"):
        os.remove("temp.zip") 

# 解壓 zip 檔案並處理圖片
def unzip_file(uploaded_zip):
    system = platform.system()
    
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        for member in zip_ref.infolist():
            if "__MACOSX" in member.filename or member.filename.startswith('.'):
                continue
            
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

def get_images_in_folder(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith('.') or os.path.isdir(os.path.join(root, file)):
                continue
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                full_image_path = os.path.join(root, file)
                relative_image_path = os.path.relpath(full_image_path, folder_path)
                image_files.append((relative_image_path, full_image_path))
    return image_files

def rename_numbers_in_folder(results):
    folders = set([result["資料夾"] for result in results])
    for folder in folders:
        folder_results = [r for r in results if r["資料夾"] == folder]
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
        folder_name = result["資料夾"]
        image_file = result["圖片"]
        new_number = result["編號"]
    
        folder_path = os.path.join("uploaded_images", folder_name)
        main_folder_path = os.path.join(folder_path, "1-Main")
        all_folder_path = os.path.join(main_folder_path, "All")
        os.makedirs(all_folder_path, exist_ok=True)
        
        old_image_path = os.path.join(folder_path, image_file)

        if new_number == "超過上限" or pd.isna(new_number):
            # 保留原檔名並移至外層資料夾
            new_image_path = os.path.join(folder_path, image_file)
        else:
            # 重命名並移至 All 資料夾
            new_image_name = f"{folder_name}_{new_number}.jpg"
            new_image_path = os.path.join(all_folder_path, new_image_name)
        
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

        if os.path.exists(old_image_path):
            os.rename(old_image_path, new_image_path)

    # 處理跳過的圖片
    for skipped_image in skipped_images:
        folder_name = skipped_image["資料夾"]
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

# 加載保存的圖片特徵
with open('image_features.pkl', 'rb') as f:
    features_by_category = pickle.load(f)

if 'file_uploader_key1' not in st.session_state:
    st.session_state['file_uploader_key1'] = 0

st.header("TP 編圖工具")
st.write("\n")

uploaded_zip = st.file_uploader(
    "上傳 zip 檔案", 
    type=["zip"], 
    key='file_uploader_' + str(st.session_state['file_uploader_key1'])
)

selectbox_placeholder = st.empty()
button_placeholder = st.empty()
if uploaded_zip:
    with selectbox_placeholder:
        selected_brand = st.selectbox(
            "請選擇品牌", 
            list(features_by_category.keys())
        )
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

    special_mappings = {}
    if selected_brand == "ADS":
        df_angles = pd.read_excel("ADS檔名角度對照表.xlsx")
        for idx, row in df_angles.iterrows():
            keyword = str(row['檔名判斷']).strip()
            category = str(row['商品分類']).strip()
            if category == 'nan' or category == '':
                category = None
            angle = str(row['對應角度']).strip()
            angles = [a.strip() for a in angle.split(',')]
            special_mappings[keyword] = {
                'category': category, 
                'angles': angles
            }

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

    # 定義組合條件
    group_conditions = [
        {
            "set_a": ['_D1_', '_D2_', '_D3_'],
            "set_b": ['_H1_', '_H2_', '_H3_']
        },
        {
            "set_a": ['_SC_'],
            "set_b": ['_Sid_Torso_']
        },
        {
            "set_a": ['_W_Model_'],
            "set_b": ['_Sid_Model_']
        }
    ]

    for folder in image_folders:
        folder_path = os.path.join("uploaded_images", folder)
        image_files = get_images_in_folder(folder_path)
        if not image_files:
            st.warning(f"資料夾 {folder} 中沒有有效的圖片，跳過此資料夾")
            continue
        folder_features = []

        progress_text.text(f"正在處理資料夾: {folder}")

        special_images = []
        folder_special_category = None

        # 初始化組合存在標誌
        group_presence = []
        for group in group_conditions:
            group_presence.append({
                "set_a_present": False,
                "set_b_present": False
            })

        # 第一次掃描：檢查檔名中是否存在組合的字串
        for image_file, image_path in image_files:
            if image_file.startswith('.') or os.path.isdir(image_path):
                continue

            for idx, group in enumerate(group_conditions):
                if any(substr in image_file for substr in group["set_a"]):
                    group_presence[idx]["set_a_present"] = True
                if any(substr in image_file for substr in group["set_b"]):
                    group_presence[idx]["set_b_present"] = True

        # 現在處理圖片
        for image_file, image_path in image_files:
            if image_file.startswith('.') or os.path.isdir(image_path):
                continue

            if any(keyword in image_file for keyword in keywords_to_skip):
                skipped_images.append({
                    "資料夾": folder, 
                    "圖片": image_file
                })
                continue

            # 檢查是否需要跳過處理
            skip_image = False
            for idx, group in enumerate(group_conditions):
                if any(substr in image_file for substr in group["set_b"]):
                    if group_presence[idx]["set_a_present"] and group_presence[idx]["set_b_present"]:
                        # 跳過處理此圖片
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
            if special_mappings:
                for substr, mapping in special_mappings.items():
                    if substr in image_file:
                        special_angles = mapping['angles']
                        special_category = mapping['category']
                        break

            if special_category and not folder_special_category:
                folder_special_category = special_category

            img = Image.open(image_path).convert('RGB')
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

        best_category = None

        if len(folder_features) == 0:
            st.warning(f"資料夾 {folder} 中沒有有效的圖片，跳過此資料夾")
            continue

        if folder_special_category:
            best_category = {
                'brand': selected_brand, 
                'category': folder_special_category
            }
        else:
            best_similarity = -1
            for img_data in folder_features:
                img_features = img_data["features"]

                for brand in features_by_category:
                    for category in features_by_category[brand]:
                        for item in features_by_category[brand][category]["labeled_features"]:
                            item_features = item["features"]
                            similarity = cosine_similarity(
                                img_features, item_features
                            )

                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_category = item["labels"]

        filtered_by_category = features_by_category[selected_brand][
            best_category["category"]
        ]["labeled_features"]

        angle_to_number = {
            item["labels"]["angle"]: item["labels"]["number"] 
            for item in filtered_by_category
        }

        used_angles = set()

        final_results = {}

        # 初始化標誌
        assigned_special_D_angle = False

        for img_data in folder_features:
            image_file = img_data["image_file"]
            special_angles = img_data["special_angles"]
            special_category = img_data["special_category"]
            img_features = img_data["features"]

            if special_angles:
                # 新增的邏輯開始
                valid_special_angles = [
                    angle for angle in special_angles 
                    if angle in angle_to_number
                ]
                if valid_special_angles:
                    if len(valid_special_angles) > 1:
                        best_angle = None
                        valid_angles_by_similarity = []
                        
                        for angle in valid_special_angles:
                            max_similarity = -1
                            for item in filtered_by_category:
                                if item["labels"]["angle"] == angle:
                                    sample_features = item["features"]
                                    similarity = cosine_similarity(
                                        img_features, sample_features
                                    )
                                    if similarity > max_similarity:
                                        max_similarity = similarity
                            
                            valid_angles_by_similarity.append(
                                (angle, max_similarity)
                            )
                        
                        valid_angles_by_similarity.sort(
                            key=lambda x: x[1], reverse=True
                        )
                        
                        for angle, similarity in valid_angles_by_similarity:
                            if angle not in ["細節", "情境細節"] and angle in used_angles:
                                pass
                            else:
                                best_angle = angle
                                best_similarity = similarity
                                break
                    
                        if best_angle:
                            used_angles.add(best_angle)
                            label_info = {
                                "資料夾": folder,
                                "圖片": image_file,
                                "商品分類": best_category["category"],
                                "角度": best_angle,
                                "編號": angle_to_number[best_angle],
                                "預測信心": f"{best_similarity * 100:.2f}%"
                            }
                            final_results[image_file] = label_info
                            # 新增檢查
                            if best_angle in ["D1", "D2", "D3"]:
                                assigned_special_D_angle = True
                        else:
                            st.warning(
                                f"圖片 '{image_file}' 沒有可用的角度可以分配"
                            )
                            final_results[image_file] = None
                    else:
                        special_angle = valid_special_angles[0]
                        if special_angle not in ["細節", "情境細節"] and special_angle in used_angles:
                            st.warning(
                                f"角度 '{special_angle}' 已被使用，圖片 '{image_file}' 無法分配"
                            )
                            final_results[image_file] = None
                        else:
                            used_angles.add(special_angle)
                            label_info = {
                                "資料夾": folder,
                                "圖片": image_file,
                                "商品分類": best_category["category"],
                                "角度": special_angle,
                                "編號": angle_to_number[special_angle],
                                "預測信心": "100.00%"
                            }
                            final_results[image_file] = label_info
                            # 新增檢查
                            if special_angle in ["D1", "D2", "D3"]:
                                assigned_special_D_angle = True
                else:
                    st.warning(
                        f"商品分類 '{best_category['category']}' 中沒有角度 '{', '.join(special_angles)}'，圖片 '{image_file}' 無法分配"
                    )
                    final_results[image_file] = None
                # 新增的邏輯結束
            else:
                final_results[image_file] = None

        non_special_images = [
            img_data for img_data in folder_features 
            if not img_data["special_angles"]
        ]

        if not special_mappings:
            non_special_images = folder_features

        image_similarity_store = {}

        for img_data in non_special_images:
            image_file = img_data["image_file"]
            if final_results.get(image_file) is not None:
                continue

            img_features = img_data["features"]
            image_similarity_list = []
            for item in filtered_by_category:
                item_angle = item["labels"]["angle"]
                # 新增條件，當已分配 D1、D2、D3 時，不再分配 "細節"
                if assigned_special_D_angle and item_angle == "細節":
                    continue
                item_features = item["features"]
                similarity = cosine_similarity(
                    img_features, item_features
                )

                image_similarity_list.append({
                    "image_file": image_file,
                    "similarity": similarity,
                    "label": item["labels"],
                    "folder": folder
                })

            image_similarity_list.sort(
                key=lambda x: x["similarity"], reverse=True
            )
            unique_labels = []
            for candidate in image_similarity_list:
                if candidate["label"]["angle"] not in [
                    label["label"]["angle"] for label in unique_labels
                ]:
                    unique_labels.append(candidate)
                if len(unique_labels) == 10:
                    break

            image_similarity_store[image_file] = unique_labels

        # 修改的邏輯開始
        unassigned_images = set(image_similarity_store.keys())

        while unassigned_images:
            angle_to_images = {}
            image_current_choices = {}
            
            for image_file in unassigned_images:
                similarity_list = image_similarity_store[image_file]
                # 找到尚未被使用的最高順位角度
                candidate = None
                for candidate_candidate in similarity_list:
                    candidate_angle = candidate_candidate["label"]["angle"]
                    if candidate_angle in ["細節", "情境細節"] or candidate_angle not in used_angles:
                        candidate = candidate_candidate
                        break
                else:
                    # 沒有可用的角度
                    candidate = None
                    candidate_angle = None
                
                if candidate:
                    candidate_angle = candidate["label"]["angle"]
                    image_current_choices[image_file] = candidate
                    if candidate_angle not in angle_to_images:
                        angle_to_images[candidate_angle] = []
                    angle_to_images[candidate_angle].append(image_file)
            
            assigned_in_this_round = set()
            for angle, images in angle_to_images.items():
                if angle in ["細節", "情境細節"]:
                    # '細節'角度可以重複使用
                    for image_file in images:
                        candidate = image_current_choices[image_file]
                        final_results[image_file] = {
                            "資料夾": candidate["folder"],
                            "圖片": image_file,
                            "商品分類": candidate["label"]["category"],
                            "角度": angle,
                            "編號": candidate["label"]["number"],
                            "預測信心": f"{candidate['similarity'] * 100:.2f}%"
                        }
                        assigned_in_this_round.add(image_file)
                elif len(images) == 1:
                    # 只有一張圖片想要這個角度，直接分配
                    image_file = images[0]
                    candidate = image_current_choices[image_file]
                    final_results[image_file] = {
                        "資料夾": candidate["folder"],
                        "圖片": image_file,
                        "商品分類": candidate["label"]["category"],
                        "角度": angle,
                        "編號": candidate["label"]["number"],
                        "預測信心": f"{candidate['similarity'] * 100:.2f}%"
                    }
                    used_angles.add(angle)
                    assigned_in_this_round.add(image_file)
                else:
                    # 多張圖片想要這個角度，選擇相似度最高的圖片
                    max_similarity = -1
                    best_image = None
                    for image_file in images:
                        candidate = image_current_choices[image_file]
                        if candidate['similarity'] > max_similarity:
                            max_similarity = candidate['similarity']
                            best_image = image_file
                    # 分配角度給相似度最高的圖片
                    candidate = image_current_choices[best_image]
                    final_results[best_image] = {
                        "資料夾": candidate["folder"],
                        "圖片": best_image,
                        "商品分類": candidate["label"]["category"],
                        "角度": angle,
                        "編號": candidate["label"]["number"],
                        "預測信心": f"{candidate['similarity'] * 100:.2f}%"
                    }
                    used_angles.add(angle)
                    assigned_in_this_round.add(best_image)
                    # 其他圖片不分配，進入下一輪

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

    results = rename_numbers_in_folder(results)

    result_df = pd.DataFrame(results)
    st.dataframe(result_df, hide_index=True, use_container_width=True)

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, index=False)
    excel_data = excel_buffer.getvalue()

    zip_data = rename_and_zip_folders(results, excel_data, skipped_images)
    
    shutil.rmtree("uploaded_images")
    os.remove("temp.zip") 
    
    if st.download_button(
        label="下載編圖結果",
        data=zip_data,
        file_name="編圖結果.zip",
        mime="application/zip",
        on_click=reset_file_uploader
    ):
        st.rerun()
