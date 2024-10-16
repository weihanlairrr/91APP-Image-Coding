import os
import requests
import shutil
from zipfile import ZipFile
import json
from pathlib import Path

# 輸入的JSON資料
json_input = '{"IH7795":"7","JR7675":"10","JR7674":"9","JR7676":"10","IF1371":"7"}'

# 根據貨號與張數產生圖片網址
def generate_urls(catalog, count):
    base_url = "https://tpimage.91app.com/adidas/"
    urls = []
    for i in range(1, int(count) + 1):
        img_num = f"{i:02d}.jpg"
        url = f"{base_url}{catalog}_OK/1-Main/All/{catalog}_{img_num}"
        urls.append(url)
    return urls

# 創建資料夾並下載圖片
def download_images(urls, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for i, url in enumerate(urls):
        img_data = requests.get(url).content
        with open(os.path.join(folder_name, f'image_{i+1}.jpg'), 'wb') as handler:
            handler.write(img_data)

# 獲取使用者的 Downloads 資料夾路徑
def get_downloads_folder():
    home = Path.home()
    downloads_folder = home / "Downloads"
    return downloads_folder

# 處理多組資料夾
def process_data(json_data):
    data = json.loads(json_data)
    
    # 找到使用者的 Downloads 資料夾
    downloads_dir = get_downloads_folder()
    
    # 建立一個總資料夾來儲存所有的資料夾
    output_dir = downloads_dir / 'all_images'
    os.makedirs(output_dir, exist_ok=True)
    
    for catalog, count in data.items():
        folder_name = output_dir / f"{catalog}"
        urls = generate_urls(catalog, count)
        
        # 下載圖片
        download_images(urls, folder_name)
        print(f"已完成 {catalog} 的圖片下載。")
    
    # 壓縮總資料夾
    zip_file = downloads_dir / 'downloaded_images.zip'
    
    with ZipFile(zip_file, 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, output_dir))
    
    # 刪除總資料夾及所有子資料夾
    shutil.rmtree(output_dir)
    print(f"已完成所有資料夾的打包並刪除臨時資料夾。")

# 處理輸入資料
process_data(json_input)
