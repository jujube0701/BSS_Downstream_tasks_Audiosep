# AudioSep 說話人識別系統

使用AudioSep作為特徵提取器，對LibriSpeech數據集進行說話人識別任務。支持一層和三層全連接層分類器的比較。

## 📁 文件結構

```
speaker_recognition/
├── config.py                           # 配置文件
├── speaker_recognition_librispeech.py  # 訓練腳本
├── speaker_recognition_inference.py    # 推理腳本
├── run_speaker_recognition.py          # 主入口腳本
├── test_system.py                      # 系統測試腳本
└── README.md                           # 說明文件
```

## 🚀 快速開始

### 1. 訓練模型（同時訓練兩個分類器）

```bash
cd AudioSep-main/speaker_recognition
python run_speaker_recognition.py train --data_dir /path/to/LibriSpeech/dev-clean
```

### 2. 推理測試

```bash
# 使用一層全連接層分類器
python run_speaker_recognition.py infer --audio_path /path/to/audio.wav --model_type linear

# 使用三層全連接層分類器
python run_speaker_recognition.py infer --audio_path /path/to/audio.wav --model_type three_layer

# 批量預測
python run_speaker_recognition.py infer --audio_path /path/to/audio/directory --model_type linear
```

## 📋 詳細參數

### 訓練參數
- `--data_dir`: LibriSpeech數據集路徑
- `--max_duration`: 音頻最大長度（秒），默認10.0
- `--epochs`: 訓練輪數，默認100
- `--lr`: 學習率，默認1e-3
- `--batch_size`: 批次大小，默認32

### 推理參數
- `--audio_path`: 音頻文件或目錄路徑
- `--model_type`: 分類器類型（linear/three_layer），默認linear
- `--device`: 設備（cuda/cpu），默認cuda

## 🔧 系統架構

```
LibriSpeech音頻 → AudioSep特徵提取器 → 512維特徵 → 分類器 → 說話人識別
                                                      ├── 一層全連接層
                                                      └── 三層全連接層
```

### 分類器架構

**一層全連接層分類器:**
```
512維特徵 → Linear(512, num_speakers) → 輸出
```

**三層全連接層分類器:**
```
512維特徵 → Linear(512, 256) → ReLU → Dropout(0.3) → 
Linear(256, 256) → ReLU → Dropout(0.3) → 
Linear(256, num_speakers) → 輸出
```

## 📊 輸出文件

訓練完成後會在項目根目錄生成：
- `librispeech_features_audiosep.pt` - 提取的特徵
- `speaker_classifier_linear.pt` - 一層全連接層分類器
- `speaker_classifier_three_layer.pt` - 三層全連接層分類器
- `librispeech_scaler.pt` - 特徵標準化器

## 💡 技術特點

- ✅ 重用AudioSep作為特徵提取器
- ✅ 同時訓練兩個不同架構的分類器
- ✅ 自動比較兩個模型的性能
- ✅ 支持GPU加速
- ✅ 自動數據預處理
- ✅ 批量推理支持

## 🏆 模型比較

訓練完成後會自動顯示兩個模型的比較結果：

```
==================================================
模型比較結果
==================================================
一層全連接層分類器 - 最佳測試準確率: XX.XX%
三層全連接層分類器 - 最佳測試準確率: XX.XX%
🏆 [表現更好的模型] 表現更好，準確率高出 X.XX%
```

## 🛠️ 故障排除

1. **CUDA內存不足**: 減少批次大小或音頻長度
2. **路徑錯誤**: 檢查數據集路徑和模型文件
3. **依賴缺失**: 確保已安裝所有必要的Python包

## 📝 使用建議

1. **選擇模型**: 
   - 一層分類器：訓練快，參數少，適合快速實驗
   - 三層分類器：表達能力強，可能獲得更高準確率

2. **性能優化**:
   - 根據GPU內存調整批次大小
   - 可以調整音頻長度來平衡速度和準確率

詳細說明請參考 `README_SpeakerRecognition.md` 