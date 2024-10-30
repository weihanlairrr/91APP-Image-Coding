# 匯入必要的函式庫
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

# 設定 Streamlit 頁面的標題和圖示
st.set_page_config(page_title='TP自動化編圖工具', page_icon='👕')

# 自訂 CSS 以調整網頁的樣式
custom_css = """
<style>
.main {
    padding-left: 28%; 
    padding-right: 28%;
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
# 將自訂的 CSS 套用到頁面中
st.markdown(custom_css, unsafe_allow_html=True)

# 設定運行裝置，優先使用 GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入 ResNet50 模型，並移除最後一層以提取特徵
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  
resnet.eval().to(device)  # 設定模型為評估模式並移至指定裝置

# 定義圖像預處理步驟
preprocess = transforms.Compose([
    transforms.Resize(256),  # 調整圖像大小到 256
    transforms.CenterCrop(224),  # 中央裁剪到 224x224
    transforms.ToTensor(),  # 轉換為張量
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # 正規化均值
        std=[0.229, 0.224, 0.225]    # 正規化標準差
    ),
])

# 定義需要跳過的關鍵字列表
keywords_to_skip = ["_SL_", "_SLB_", "_SMC_", "_Fout_", "-1", "_Sid_", "_BL_", "_FM_", "_BSM_", "_LSL_", "Thumbs","_Bex_"]

def get_image_features(image, model):
    """
    提取圖像的特徵向量。
    
    參數:
    image (PIL.Image): 要處理的圖像。
    model (torch.nn.Module): 用於提取特徵的模型。
    
    返回:
    numpy.ndarray: 圖像的特徵向量。
    """
    image = preprocess(image).unsqueeze(0).to(device)  # 對圖像進行預處理並添加批次維度
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()  # 提取特徵並轉換為 NumPy 陣列
    return features

def cosine_similarity(a, b):
    """
    計算兩個向量的餘弦相似度。
    
    參數:
    a (numpy.ndarray): 向量 a。
    b (numpy.ndarray): 向量 b。
    
    返回:
    float: 餘弦相似度。
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def reset_file_uploader():
    """
    重設文件上傳器的狀態，並刪除暫存的上傳資料夾和壓縮檔。
    """
    st.session_state['file_uploader_key1'] += 1
    if os.path.exists("uploaded_images"):
        shutil.rmtree("uploaded_images")  # 刪除上傳的圖片資料夾
    if os.path.exists("temp.zip"):
        os.remove("temp.zip")  # 刪除暫存的壓縮檔

def unzip_file(uploaded_zip):
    """
    解壓上傳的 zip 檔案到指定的資料夾。
    
    參數:
    uploaded_zip (str): 上傳的 zip 檔案路徑。
    """
    system = platform.system()  # 獲取作業系統類型
    
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        for member in zip_ref.infolist():
            # 跳過 __MACOSX 資料夾和隱藏文件
            if "__MACOSX" in member.filename or member.filename.startswith('.'):
                continue
            
            # 根據不同作業系統處理檔名編碼
            if system == "Windows":
                try:
                    member.filename = member.filename.encode('utf-8').decode('utf-8')
                except UnicodeDecodeError:
                    member.filename = member.filename.encode('utf-8').decode('latin1')
            elif system == "Darwin":  # macOS
                try:
                    member.filename = member.filename.encode('cp437').decode('utf-8')
                except UnicodeDecodeError:
                    member.filename = member.filename.encode('cp437').decode('latin1')
            else:
                try:
                    member.filename = member.filename.encode('utf-8').decode('utf-8')
                except UnicodeDecodeError:
                    member.filename = member.filename.encode('utf-8').decode('latin1')
            
            # 解壓檔案到 "uploaded_images" 資料夾
            zip_ref.extract(member, "uploaded_images")

def get_images_in_folder(folder_path):
    """
    獲取指定資料夾內所有有效的圖片檔案。
    
    參數:
    folder_path (str): 資料夾路徑。
    
    返回:
    list: 包含圖片相對路徑和完整路徑的元組列表。
    """
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 跳過以 '.' 開頭的檔案和子資料夾
            if file.startswith('.') or os.path.isdir(os.path.join(root, file)):
                continue
            # 檢查檔案是否為圖片格式
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                full_image_path = os.path.join(root, file)
                relative_image_path = os.path.relpath(full_image_path, folder_path)
                image_files.append((relative_image_path, full_image_path))
    return image_files

def rename_numbers_in_folder(results):
    """
    重新編號資料夾內的圖片，並處理超過上限的情況。
    
    參數:
    results (list): 包含圖片處理結果的字典列表。
    
    返回:
    list: 更新後的結果列表。
    """
    folders = set([result["資料夾"] for result in results])  # 獲取所有資料夾名稱
    for folder in folders:
        folder_results = [r for r in results if r["資料夾"] == folder]
        # 檢查是否有未編號的圖片
        if any(pd.isna(r["編號"]) or r["編號"] == "" for r in folder_results):
            continue
        # 按編號排序
        folder_results.sort(key=lambda x: int(x["編號"]))
        for idx, result in enumerate(folder_results):
            if idx < 10:
                result["編號"] = f'{idx+1:02}'  # 重新編號為 01, 02, ...
            else:
                result["編號"] = "超過上限"  # 超過上限的編號標記
    return results

def rename_and_zip_folders(results, output_excel_data, skipped_images):
    """
    重新命名並壓縮資料夾內的圖片，並將結果保存為壓縮檔。
    
    參數:
    results (list): 包含圖片處理結果的字典列表。
    output_excel_data (bytes): Excel 檔案的二進位資料。
    skipped_images (list): 被跳過的圖片列表。
    
    返回:
    bytes: 壓縮檔的二進位資料。
    """
    for result in results:
        folder_name = result["資料夾"]
        image_file = result["圖片"]
        new_number = result["編號"]
    
        folder_path = os.path.join("uploaded_images", folder_name)
        main_folder_path = os.path.join(folder_path, "1-Main")
        all_folder_path = os.path.join(main_folder_path, "All")
        os.makedirs(all_folder_path, exist_ok=True)  # 創建主資料夾和 All 子資料夾
        
        old_image_path = os.path.join(folder_path, image_file)

        if new_number == "超過上限" or pd.isna(new_number):
            new_image_path = os.path.join(folder_path, image_file)  # 保持原名
        else:
            new_image_name = f"{folder_name}_{new_number}.jpg"  # 新的圖片名稱
            new_image_path = os.path.join(all_folder_path, new_image_name)

        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

        if os.path.exists(old_image_path):
            os.rename(old_image_path, new_image_path)  # 重新命名圖片

    # 處理被跳過的圖片
    for skipped_image in skipped_images:
        folder_name = skipped_image["資料夾"]
        image_file = skipped_image["圖片"]
        folder_path = os.path.join("uploaded_images", folder_name)
        old_image_path = os.path.join(folder_path, image_file)
        
        if os.path.exists(old_image_path):
            new_image_path = os.path.join(folder_path, image_file)
            os.rename(old_image_path, new_image_path)

    # 創建壓縮檔
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder in os.listdir("uploaded_images"):
            folder_path = os.path.join("uploaded_images", folder)
            if os.path.isdir(folder_path):
                new_folder_name = f"{folder}_OK"
                new_folder_path = os.path.join("uploaded_images", new_folder_name)
                os.rename(folder_path, new_folder_path)  # 重新命名資料夾
                
                for root, dirs, files in os.walk(new_folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 將檔案添加到壓縮檔中，保持相對路徑
                        zipf.write(file_path, os.path.relpath(file_path, "uploaded_images"))
        
        # 將 Excel 結果添加到壓縮檔
        zipf.writestr("編圖結果.xlsx", output_excel_data)

    return zip_buffer.getvalue()  # 返回壓縮檔的二進位資料

# 載入已預先計算的圖像特徵資料
with open('image_features.pkl', 'rb') as f:
    features_by_category = pickle.load(f)

# 初始化文件上傳器的狀態
if 'file_uploader_key1' not in st.session_state:
    st.session_state['file_uploader_key1'] = 0

# 設定頁面的標題
st.header("TP 編圖工具")
st.write("\n")

# 文件上傳器，允許使用者上傳 zip 檔案
uploaded_zip = st.file_uploader(
    "上傳 zip 檔案", 
    type=["zip"], 
    key='file_uploader_' + str(st.session_state['file_uploader_key1'])
)

# 建立佔位符，用於後續動態添加選項框和按鈕
selectbox_placeholder = st.empty()
button_placeholder = st.empty()

# 如果有上傳檔案，顯示品牌選擇框和開始按鈕
if uploaded_zip:
    with selectbox_placeholder:
        selected_brand = st.selectbox(
            "請選擇品牌", 
            list(features_by_category.keys())  # 品牌列表來源於已載入的特徵資料
        )
    with button_placeholder:
        start_running = st.button("開始執行")

# 如果檔案已上傳且使用者按下開始執行按鈕
if uploaded_zip and start_running:
    selectbox_placeholder.empty()  # 清空品牌選擇框
    button_placeholder.empty()      # 清空按鈕
    st.write("\n")
    
    # 如果存在之前上傳的圖片資料夾，刪除它
    if os.path.exists("uploaded_images"):
        shutil.rmtree("uploaded_images")
        
    # 將上傳的 zip 檔案寫入暫存檔案
    with open("temp.zip", "wb") as f:
        f.write(uploaded_zip.getbuffer())

    # 解壓上傳的 zip 檔案
    unzip_file("temp.zip")

    # 定義特殊的映射關係
    special_mappings = {}
    if selected_brand == "ADS":
        # 讀取 ADS 品牌的檔名與角度對照表
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

    # 獲取所有上傳的圖片資料夾
    image_folders = [
        f for f in os.listdir("uploaded_images") 
        if os.path.isdir(os.path.join("uploaded_images", f)) 
        and not f.startswith('__MACOSX') and not f.startswith('.')
    ]
    results = []        # 儲存處理結果
    skipped_images = [] # 儲存被跳過的圖片
    progress_bar = st.progress(0)  # 建立進度條
    progress_text = st.empty()      # 建立進度文字

    total_folders = len(image_folders)  # 總共需要處理的資料夾數
    processed_folders = 0               # 已處理的資料夾數

    # 定義分組條件，用於判斷圖片的類型
    group_conditions = [
        {
            "set_a": ['_D1_', '_D2_', '_D3_', '_D4_', '_D5_'],
            "set_b": ['_H1_', '_H2_', '_H3_','_H4_','_H5_']
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

    # 逐一處理每個圖片資料夾
    for folder in image_folders:
        folder_path = os.path.join("uploaded_images", folder)
        image_files = get_images_in_folder(folder_path)  # 獲取資料夾內的所有圖片
        if not image_files:
            st.warning(f"資料夾 {folder} 中沒有有效的圖片，跳過此資料夾")
            continue  # 跳過沒有圖片的資料夾
        folder_features = []  # 儲存資料夾內所有圖片的特徵
        progress_text.text(f"正在處理資料夾: {folder}")  # 更新進度文字
        special_images = []  # 儲存特殊角度的圖片
        folder_special_category = None  # 資料夾的特殊商品分類

        # 初始化分組存在情況的列表
        group_presence = []
        for group in group_conditions:
            group_presence.append({
                "set_a_present": False,
                "set_b_present": False
            })

        # 檢查每個資料夾中各分組條件是否存在
        for image_file, image_path in image_files:
            if image_file.startswith('.') or os.path.isdir(image_path):
                continue

            for idx, group in enumerate(group_conditions):
                if any(substr in image_file for substr in group["set_a"]):
                    group_presence[idx]["set_a_present"] = True
                if any(substr in image_file for substr in group["set_b"]):
                    group_presence[idx]["set_b_present"] = True

        # 逐一處理每張圖片
        for image_file, image_path in image_files:
            if image_file.startswith('.') or os.path.isdir(image_path):
                continue

            # 如果圖片名稱包含需要跳過的關鍵字，則加入跳過列表
            if any(keyword in image_file for keyword in keywords_to_skip):
                skipped_images.append({
                    "資料夾": folder, 
                    "圖片": image_file
                })
                continue

            skip_image = False
            # 根據分組條件判斷是否需要跳過圖片
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
            # 根據特殊映射關係判斷圖片的特殊角度和分類
            if special_mappings:
                for substr, mapping in special_mappings.items():
                    if substr in image_file:
                        special_angles = mapping['angles']
                        special_category = mapping['category']
                        break

            # 如果有特殊分類，則設定資料夾的特殊分類
            if special_category and not folder_special_category:
                folder_special_category = special_category

            # 開啟圖片並提取特徵
            img = Image.open(image_path).convert('RGB')
            img_features = get_image_features(img, resnet)
            folder_features.append({
                "image_file": image_file,
                "features": img_features,
                "special_angles": special_angles,
                "special_category": special_category
            })

            # 如果有特殊角度，則記錄該圖片
            if special_angles:
                special_images.append({
                    "image_file": image_file,
                    "special_angles": special_angles
                })

        best_category = None  # 最佳匹配的商品分類

        # 如果資料夾內沒有有效的圖片，跳過該資料夾
        if len(folder_features) == 0:
            st.warning(f"資料夾 {folder} 中沒有有效的圖片，跳過此資料夾")
            continue

        # 如果有特殊分類，則直接使用特殊分類
        if folder_special_category:
            best_category = {
                'brand': selected_brand, 
                'category': folder_special_category
            }
        else:
            # 否則，根據相似度選擇最佳分類
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

        # 根據最佳分類過濾對應的特徵
        filtered_by_category = features_by_category[selected_brand][
            best_category["category"]
        ]["labeled_features"]

        # 建立角度到編號的對應字典
        angle_to_number = {
            item["labels"]["angle"]: item["labels"]["number"] 
            for item in filtered_by_category
        }

        used_angles = set()  # 已使用的角度集合
        final_results = {}    # 最終的分配結果
        assigned_special_D_angle = False  # 是否已分配特殊的 D 角度

        # 逐一處理資料夾內的圖片
        for img_data in folder_features:
            image_file = img_data["image_file"]
            special_angles = img_data["special_angles"]
            special_category = img_data["special_category"]
            img_features = img_data["features"]

            # 如果圖片有特殊角度
            if special_angles:
                # 過濾出有效的特殊角度
                valid_special_angles = [
                    angle for angle in special_angles 
                    if angle in angle_to_number
                ]
                if valid_special_angles:
                    if len(valid_special_angles) > 1:
                        best_angle = None
                        valid_angles_by_similarity = []
                        
                        # 根據相似度排序有效的角度
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
                        
                        # 按相似度降序排序角度
                        valid_angles_by_similarity.sort(
                            key=lambda x: x[1], reverse=True
                        )
                        
                        # 選擇最佳角度，避免重複使用特定角度
                        for angle, similarity in valid_angles_by_similarity:
                            if angle not in ["細節", "情境細節","情境帽子配戴照"] and angle in used_angles:
                                pass
                            else:
                                best_angle = angle
                                best_similarity = similarity
                                break
                    
                        if best_angle:
                            used_angles.add(best_angle)  # 標記角度已使用
                            label_info = {
                                "資料夾": folder,
                                "圖片": image_file,
                                "商品分類": best_category["category"],
                                "角度": best_angle,
                                "編號": angle_to_number[best_angle],
                                "預測信心": f"{best_similarity * 100:.2f}%"
                            }
                            final_results[image_file] = label_info
                            if best_angle in ["D1", "D2", "D3", "D4", "D5",'_H1_', '_H2_', '_H3_','_H4_','_H5_']:
                                assigned_special_D_angle = True
                        else:
                            st.warning(
                                f"圖片 '{image_file}' 沒有可用的角度可以分配"
                            )
                            final_results[image_file] = None
                    else:
                        # 僅有一個特殊角度的情況
                        special_angle = valid_special_angles[0]
                        if special_angle not in ["細節", "情境細節","情境帽子配戴照"] and special_angle in used_angles:
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
                            if special_angle in ["D1", "D2", "D3", "D4", "D5",'_H1_', '_H2_', '_H3_','_H4_','_H5_']:
                                assigned_special_D_angle = True
                else:
                    # 如果特殊角度無效，顯示警告
                    st.warning(
                        f"商品分類 '{best_category['category']}' 中沒有角度 '{', '.join(special_angles)}'，圖片 '{image_file}' 無法分配"
                    )
                    final_results[image_file] = None
            else:
                final_results[image_file] = None  # 沒有特殊角度的圖片暫時不處理

        # 獲取所有非特殊角度的圖片
        non_special_images = [
            img_data for img_data in folder_features 
            if not img_data["special_angles"]
        ]

        # 如果沒有特殊映射，所有圖片都視為非特殊角度
        if not special_mappings:
            non_special_images = folder_features

        image_similarity_store = {}  # 儲存圖片的相似度列表

        # 逐一計算非特殊圖片的相似度
        for img_data in non_special_images:
            image_file = img_data["image_file"]
            if final_results.get(image_file) is not None:
                continue  # 已經有分配結果的圖片跳過

            img_features = img_data["features"]
            image_similarity_list = []
            for item in filtered_by_category:
                item_angle = item["labels"]["angle"]
                if assigned_special_D_angle and item_angle == "細節":
                    continue  # 如果已分配特殊 D 角度，跳過細節角度
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

            # 按相似度降序排序
            image_similarity_list.sort(
                key=lambda x: x["similarity"], reverse=True
            )
            unique_labels = []
            # 選取前 10 個不重複的角度標籤
            for candidate in image_similarity_list:
                if candidate["label"]["angle"] not in [
                    label["label"]["angle"] for label in unique_labels
                ]:
                    unique_labels.append(candidate)
                if len(unique_labels) == 10:
                    break

            image_similarity_store[image_file] = unique_labels

        unassigned_images = set(image_similarity_store.keys())  # 尚未分配的圖片集合

        # 持續分配直到所有圖片都被處理或無法進一步分配
        while unassigned_images:
            angle_to_images = {}
            image_current_choices = {}
            
            for image_file in unassigned_images:
                similarity_list = image_similarity_store[image_file]
                candidate = None
                for candidate_candidate in similarity_list:
                    candidate_angle = candidate_candidate["label"]["angle"]
                    if candidate_angle in ["細節", "情境細節","情境帽子配戴照"] or candidate_angle not in used_angles:
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
            for angle, images in angle_to_images.items():
                if angle in ["細節", "情境細節","情境帽子配戴照"]:
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
                    # 只有一張圖片符合此角度
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
                    used_angles.add(angle)  # 標記角度已使用
                    assigned_in_this_round.add(image_file)
                else:
                    # 多張圖片符合此角度，選擇相似度最高的一張
                    max_similarity = -1
                    best_image = None
                    for image_file in images:
                        candidate = image_current_choices[image_file]
                        if candidate['similarity'] > max_similarity:
                            max_similarity = candidate['similarity']
                            best_image = image_file
                    candidate = image_current_choices[best_image]
                    final_results[best_image] = {
                        "資料夾": candidate["folder"],
                        "圖片": best_image,
                        "商品分類": candidate["label"]["category"],
                        "角度": angle,
                        "編號": candidate["label"]["number"],
                        "預測信心": f"{candidate['similarity'] * 100:.2f}%"
                    }
                    used_angles.add(angle)  # 標記角度已使用
                    assigned_in_this_round.add(best_image)

            unassigned_images -= assigned_in_this_round  # 移除已分配的圖片
            if not assigned_in_this_round:
                break  # 如果本輪沒有分配，則結束迴圈

        # 將最終分配結果加入結果列表
        for image_file, assignment in final_results.items():
            if assignment is not None:
                results.append(assignment)

        processed_folders += 1
        progress_bar.progress(processed_folders / total_folders)  # 更新進度條

    # 清空進度條和進度文字
    progress_bar.empty()
    progress_text.empty()

    # 重新編號資料夾內的圖片
    results = rename_numbers_in_folder(results)

    # 將結果轉換為 DataFrame 並顯示在網頁上
    result_df = pd.DataFrame(results)
    st.dataframe(result_df, hide_index=True, use_container_width=True)

    # 將結果保存為 Excel 檔案
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, index=False)
    excel_data = excel_buffer.getvalue()

    # 重新命名並壓縮資料夾，包含 Excel 結果和跳過的圖片
    zip_data = rename_and_zip_folders(results, excel_data, skipped_images)
    
    # 清理暫存資料夾和檔案
    shutil.rmtree("uploaded_images")
    os.remove("temp.zip") 
    
    # 提供下載按鈕讓使用者下載結果壓縮檔
    if st.download_button(
        label="下載編圖結果",
        data=zip_data,
        file_name="編圖結果.zip",
        mime="application/zip",
        on_click=reset_file_uploader  # 下載後重設上傳器
    ):
        st.rerun()  # 重新執行頁面以清除狀態
