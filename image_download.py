import os
import requests
import shutil
from zipfile import ZipFile
import json

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

# 處理多組資料夾
def process_data(json_data):
    data = json.loads(json_data)
    for catalog, count in data.items():
        folder_name = f"{catalog}"
        urls = generate_urls(catalog, count)
        
        # 下載圖片
        download_images(urls, folder_name)
        
        # 壓縮資料夾
        output_dir = 'C:/Users/albertlai/Downloads'
        os.makedirs(output_dir, exist_ok=True)
        zip_file = os.path.join(output_dir, f'{folder_name}.zip')
        
        with ZipFile(zip_file, 'w') as zipf:
            for root, dirs, files in os.walk(folder_name):
                for file in files:
                    zipf.write(os.path.join(root, file))
        
        # 刪除臨時資料夾
        shutil.rmtree(folder_name)
        print(f"已完成 {catalog} 的圖片下載、壓縮，並刪除臨時資料夾。")

# 處理輸入資料
process_data(json_input)
