# AudioSep 說話人識別下游任務

這個項目使用AudioSep作為特徵提取器，對LibriSpeech數據集進行說話人識別任務。

## 系統架構

```
AudioSep (特徵提取器)
    ↓ (512維特徵)
全連接層分類器
    ↓
說話人識別結果
```

## 文件說明

- `speaker_recognition_librispeech.py`: 主要的訓練腳本
- `speaker_recognition_inference.py`: 推理腳本
- `config.py`: 配置文件（已更新）

## 使用步驟

### 1. 準備LibriSpeech數據集

下載LibriSpeech數據集並解壓到指定目錄：
```bash
# 例如下載LibriSpeech-100h
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz
```

### 2. 特徵提取和訓練

```bash
python speaker_recognition_librispeech.py \
    --data_dir /path/to/LibriSpeech/dev-clean \
    --max_duration 10.0 \
    --epochs 100 \
    --lr 1e-3 \
    --batch_size 32
```

參數說明：
- `--data_dir`: LibriSpeech數據集路徑
- `--max_duration`: 音頻最大長度（秒）
- `--epochs`: 訓練輪數
- `--lr`: 學習率
- `--batch_size`: 批次大小

### 3. 推理測試

單個文件預測：
```bash
python speaker_recognition_inference.py --audio_path /path/to/audio.wav
```

批量預測：
```bash
python speaker_recognition_inference.py --audio_path /path/to/audio/directory
```

## 輸出文件

訓練完成後會生成以下文件：

- `librispeech_features_audiosep.pt`: 提取的音頻特徵
- `speaker_classifier_librispeech.pt`: 訓練好的分類器
- `librispeech_scaler.pt`: 特徵標準化器

## 技術細節

### 特徵提取
- 使用AudioSep的CLAP編碼器提取512維音頻特徵
- 音頻預處理：重採樣到32kHz，轉單聲道，截斷/填充到固定長度

### 分類器
- 單層全連接網絡：512維 → 說話人數量
- 使用CrossEntropyLoss損失函數
- Adam優化器

### 數據處理
- 自動掃描LibriSpeech目錄結構
- 從文件路徑提取說話人ID
- 80/20訓練測試集分割

## 性能優化建議

1. **GPU加速**: 確保有足夠的GPU內存
2. **批次大小**: 根據GPU內存調整批次大小
3. **音頻長度**: 可以調整max_duration來平衡速度和準確率
4. **數據增強**: 可以添加音頻增強技術提高魯棒性

## 擴展功能

### 支持其他數據集
可以修改`LibriSpeechDataset`類來支持其他說話人識別數據集：

```python
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        # 實現自定義數據集載入邏輯
        pass
```

### 多層分類器
可以修改`SpeakerClassifier`來使用更複雜的網絡架構：

```python
class MultiLayerSpeakerClassifier(nn.Module):
    def __init__(self, feature_dim=512, num_speakers=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_speakers)
        )
    
    def forward(self, x):
        return self.net(x)
```

## 故障排除

### 常見問題

1. **CUDA內存不足**
   - 減少批次大小
   - 減少音頻最大長度
   - 使用CPU訓練（較慢）

2. **數據集路徑錯誤**
   - 確保LibriSpeech目錄結構正確
   - 檢查文件權限

3. **模型文件缺失**
   - 確保已下載AudioSep預訓練模型
   - 檢查配置文件中的路徑

### 調試模式

可以添加詳細的日誌輸出來調試問題：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 參考文獻

- AudioSep: https://github.com/AudioSep/AudioSep
- LibriSpeech: https://www.openslr.org/12/
- CLAP: https://github.com/LAION-AI/CLAP 