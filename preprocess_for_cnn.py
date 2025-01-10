import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa

DATA_PATH = "data/"  # 音訊資料的根目錄
FEATURE_PATH = "features/"  # 特徵檔案儲存路徑

# 儲存音訊特徵到 .npy 檔案
def save_data_to_array(n_mfcc=20):
    """
    提取音訊檔案的 MFCC 特徵，並將特徵和標籤儲存為 .npy 檔案
    """
    if not os.path.exists(FEATURE_PATH):
        os.makedirs(FEATURE_PATH)

    for label in os.listdir(DATA_PATH):
        label_path = os.path.join(DATA_PATH, label)

        if not os.path.isdir(label_path):
            continue

        mfcc_features = []  # 初始化特徵列表

        print(f"Processing label: {label}")

        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                try:
                    # 讀取音訊
                    audio, sr = librosa.load(file_path, sr=16000)
                    # 提取 MFCC 特徵
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
                    if mfcc.shape[1] < 1:  # 確保特徵有效
                        print(f"Invalid MFCC shape for file: {file_path}")
                        continue

                    # 平均化特徵
                    mfcc_scaled = np.mean(mfcc.T, axis=0)
                    mfcc_features.append(mfcc_scaled)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        # 確保特徵列表內的所有元素形狀一致
        try:
            mfcc_features = np.array(mfcc_features)
            print(f"Saving features for label {label}, Shape: {mfcc_features.shape}")
            np.save(os.path.join(FEATURE_PATH, f"{label}_features.npy"), mfcc_features)
        except ValueError as e:
            print(f"Error saving features for label {label}: {e}")

# 載入特徵資料並分割成訓練集和測試集
def get_train_test(test_size=0.2):
    """
    載入儲存的特徵數據，並將其分割為訓練集和測試集
    """
    features = []
    labels = []

    # 讀取每個類別的特徵檔案
    for file in os.listdir(FEATURE_PATH):
        if file.endswith("_features.npy"):
            label = file.split("_features.npy")[0]
            data = np.load(os.path.join(FEATURE_PATH, file))

            features.extend(data)
            labels.extend([label] * len(data))  # 每個特徵對應的標籤

    # 將標籤轉換為數字編碼
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # 將特徵轉換為 numpy 陣列
    features = np.array(features)

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=test_size, random_state=42)
    print(f"X_train.shape={X_train.shape}, X_test.shape={X_test.shape}")
    return X_train, X_test, y_train, y_test
