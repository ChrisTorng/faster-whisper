import os
import sys
from faster_whisper import WhisperModel

# model_size = "large-v3"
# model_size = "tiny"
model_size = "turbo"

# 檢查命令列參數
if len(sys.argv) < 2:
    print("用法: python transcribe.py <音檔路徑>")
    sys.exit(1)

# 從命令列參數獲取輸入音檔路徑
audio_file = sys.argv[1]

# 確認檔案存在
if not os.path.exists(audio_file):
    print(f"錯誤: 無法找到檔案 '{audio_file}'")
    sys.exit(1)

# 準備輸出檔案路徑
base_name = os.path.splitext(audio_file)[0]
output_file = base_name + ".srt"
txt_output_file = base_name + ".txt"  # 新增 txt 輸出檔案路徑

# 檢查檔案是否已存在，若存在則附加數字
counter = 1
while os.path.exists(output_file):
    output_file = f"{base_name}_{counter}.srt"
    counter += 1

# 同樣處理 txt 輸出檔案
counter = 1
while os.path.exists(txt_output_file):
    txt_output_file = f"{base_name}_{counter}.txt"
    counter += 1

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

print(f"處理音檔: {audio_file}")
print(f"轉錄結果將儲存至: {output_file} 及 {txt_output_file}")

segments, info = model.transcribe(audio_file, beam_size=5, language="zh", initial_prompt="台灣繁體中文")

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# 將秒數轉換為 SRT 格式的時間碼 (小時:分鐘:秒,毫秒)
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"

# 建立檔案並寫入 SRT/TXT 格式的轉錄結果
with open(output_file, "w", encoding="utf-8") as fsrt:
    with open(txt_output_file, "w", encoding="utf-8") as ftxt:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            
            srt_entry = f"{i}\n{start_time} --> {end_time}\n{segment.text.strip()}\n"
            fsrt.write(srt_entry + "\n")

            txt_entry = f"{start_time} {end_time} {segment.text.strip()}"
            print(txt_entry)
            ftxt.write(txt_entry + "\n")

print(f"轉錄完成! 結果已儲存至:\n- SRT: {output_file}\n- TXT: {txt_output_file}")
