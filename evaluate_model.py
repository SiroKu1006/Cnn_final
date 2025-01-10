import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from main import ASRNet  # 引用 main.py 中的 ASRNet 模型

def load_model(model_path: str, device: torch.device):
    """
    載入儲存的模型權重並初始化模型。
    :param model_path: 模型檔案的路徑 (如 'best_model_10.pth')。
    :param device: 運行設備 (如 'cpu' 或 'cuda')。
    :return: 初始化並載入權重的模型。
    """
    model = ASRNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 設置為評估模式
    return model


def preprocess_audio(audio_file, device, target_num_frames=100):
    """
    預處理音訊檔案，生成 MFCC 特徵。
    :param audio_file: 音訊檔案路徑。
    :param device: 運行設備。
    :param target_num_frames: 最終的時間幀數。
    :return: 處理後的 MFCC 特徵張量。
    """
    waveform, sample_rate = torchaudio.load(audio_file)

    # 如果有多個通道，轉換為單通道 (取平均值)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 若採樣率非 16000，則重採樣
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    mfcc_transform = torchaudio.transforms.MFCC(
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
    )
    mfcc = mfcc_transform(waveform)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

    # 時間維度截斷或補零至 target_num_frames
    time_frames = mfcc.shape[2]
    if time_frames > target_num_frames:
        mfcc = mfcc[:, :, :target_num_frames]
    else:
        pad_amount = target_num_frames - time_frames
        mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount))

    return mfcc.to(device)



def evaluate_model(model, test_data_dir, device):
    """
    使用模型對測試資料進行評估，並生成混淆矩陣（以百分比表示）。
    :param model: 已載入的模型。
    :param test_data_dir: 測試資料目錄，包含 .wav 檔案。
    :param device: 運行設備。
    """
    true_labels = []
    predicted_labels = []

    # 遍歷主資料夾及其子資料夾
    for folder_name in os.listdir(test_data_dir):
        folder_path = os.path.join(test_data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # 忽略非資料夾的內容

        # 遍歷資料夾中的音訊檔案
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".wav"):
                # 提取檔案名稱中的真實標籤 a 和序號 b
                parts = file_name.split("_")
                if len(parts) != 2 or not parts[1].lower().endswith(".wav"):
                    print(f"跳過無效檔案: {file_name}")
                    continue

                try:
                    true_label = int(parts[0])  # a 為真實標籤
                except ValueError:
                    print(f"跳過無效檔案: {file_name}")
                    continue

                audio_file = os.path.join(folder_path, file_name)

                # 預處理音訊並進行推論
                mfcc = preprocess_audio(audio_file, device)
                mfcc = mfcc.unsqueeze(0)  # 增加 batch 維度
                with torch.no_grad():
                    output = model(mfcc)
                    predicted_label = output.argmax(dim=1).item()
                # 記錄真實標籤與預測標籤
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)

    # 生成混淆矩陣
    cm = confusion_matrix(true_labels, predicted_labels, labels=np.arange(10))
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 轉換為百分比
    print(f"辨識正確率為: {np.trace(cm) / np.sum(cm) * 100:.2f}%")
    # 繪製混淆矩陣
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=np.arange(10))
    disp.plot(cmap="Blues", xticks_rotation='vertical')

    plt.title("Confusion Matrix (Percentage)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
    




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_model_10.pth"
    test_data_dir = "test_audio_cnn"  # 測試資料目錄

    # 載入模型
    model = load_model(model_path, device)
    evaluate_model(model, test_data_dir, device)
