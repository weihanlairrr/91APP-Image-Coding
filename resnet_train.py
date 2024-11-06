import pickle

# 特徵檔案路徑
feature_file_path = r'C:\Users\albertlai\Desktop\python\91APP-AI-Image-Coding\image_features.pkl'

# 記錄刪除的重複資料數量
deleted_count = 0
deleted_by_category = {}  # 用於記錄每個分類被刪除的數量

def load_features(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# 移除重複特徵的函數（特徵向量和標籤都需相同）
def remove_duplicates(features_by_category):
    global deleted_count, deleted_by_category
    for brand, categories in features_by_category.items():
        for category, data in categories.items():
            # 使用 set() 記錄唯一的 (特徵, 標籤) 組合，便於比對
            unique_entries = set()
            unique_labeled_features = []

            for item in data["labeled_features"]:
                # 創建 (特徵, 標籤) 的唯一鍵
                feature_tuple = (
                    tuple(item["features"]),  # 特徵向量
                    item["labels"]["brand"],   # 品牌
                    item["labels"]["category"],# 商品分類
                    item["labels"]["angle"]    # 角度
                )

                if feature_tuple not in unique_entries:
                    unique_entries.add(feature_tuple)
                    unique_labeled_features.append(item)
                else:
                    deleted_count += 1  # 若重複，刪除並計數
                    if category not in deleted_by_category:
                        deleted_by_category[category] = 0
                    deleted_by_category[category] += 1  # 記錄該分類的刪除數量

            # 更新去重後的特徵列表
            features_by_category[brand][category]["labeled_features"] = unique_labeled_features
    return features_by_category

# 載入和去除重複資料
features_by_category = load_features(feature_file_path)
features_by_category = remove_duplicates(features_by_category)

# 儲存去重後的資料
with open(feature_file_path, 'wb') as f:
    pickle.dump(features_by_category, f)

# 列出刪除結果
print(f"總共刪除了 {deleted_count} 筆重複資料")
for category, count in deleted_by_category.items():
    print(f"{category}: {count} 筆")
