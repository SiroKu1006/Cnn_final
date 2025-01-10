import os

def swap_wav_filenames(directory):
    """
    將指定資料夾中的 .wav 檔案名稱進行對調，例如 a_b.wav 變成 b_a.wav。
    :param directory: 資料夾路徑。
    """
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            # 檢查檔名格式是否正確
            parts = filename.split("_")
            if len(parts) != 2 or not parts[1].lower().endswith(".wav"):
                print(f"跳過無效檔案: {filename}")
                continue

            # 提取 a 和 b，並對調名稱
            part1 = parts[0]
            part2 = parts[1].replace(".wav", "")
            new_filename = f"{part2}_{part1}.wav"

            # 重新命名檔案
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            try:
                os.rename(old_path, new_path)
                print(f"已重新命名: {filename} -> {new_filename}")
            except Exception as e:
                print(f"重新命名失敗: {filename}, 錯誤: {e}")

# 執行程式
if __name__ == "__main__":
    directory = "t/a"
    if os.path.isdir(directory):
        swap_wav_filenames(directory)
    else:
        print("提供的路徑不是有效的資料夾。")
