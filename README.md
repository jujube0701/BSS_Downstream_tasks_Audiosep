# AudioSep 下游任務評估專案

這個專案系統性地評估了 AudioSep 音訊盲源分離模型在6個不同下游任務上的表現，包括環境音分類、醫學音頻分類、城市聲音分類、樂器分類、說話人識別和關鍵詞檢測。

## 📁 專案結構

```
AudioSep/
├── esc50_feature_fc/           # ESC-50 環境音分類
│   ├── config.py              # 配置檔案
│   ├── extract_esc50_features_audiosep.py  # 特徵提取腳本
│   ├── esc50_linear_classifier.py          # 單層分類器訓練
│   ├── esc50_features_audiosep.pt          # 提取的特徵
│   ├── esc50_linear_classifier.pt          # 單層分類器模型
│   └── esc50_three_classifier.pt           # 三層分類器模型
│
├── icbhi_feature_fc/           # ICBHI 醫學音頻分類
│   ├── extract_icbhi_features_audiosep.py  # 特徵提取腳本
│   ├── icbhi_linear_classifier.py          # 單層分類器訓練
│   ├── icbhi_features_audiosep.pt          # 提取的特徵
│   ├── icbhi_linear_classifier.pt          # 單層分類器模型
│   ├── icbhi_three_classifier.pt           # 三層分類器模型
│   └── result.csv                          # 分類結果
│
├── urbansound8k_feature_fc/    # UrbanSound8K 城市聲音分類
│   ├── extract_urbansound8k_features_audiosep.py  # 特徵提取腳本
│   ├── urbansound8k_linear_classifier.py          # 單層分類器訓練
│   ├── urbansound8k_features_audiosep.pt          # 提取的特徵
│   ├── urbansound8k_linear_classifier.pt          # 單層分類器模型
│   └── urbansound8k_three_classifier.pt           # 三層分類器模型
│
├── nsynth_feature_fc/          # NSynth 樂器分類
│   ├── config.py              # 配置檔案
│   ├── extract_nsynth_train_features_audiosep.py  # 訓練集特徵提取
│   ├── extract_nsynth_valid_features_audiosep.py  # 驗證集特徵提取
│   ├── extract_nsynth_test_features_audiosep.py   # 測試集特徵提取
│   ├── train_nsynth_fc_classifiers_valid_test.py  # 分類器訓練
│   ├── nsynth_train_features.pt            # 訓練集特徵
│   ├── nsynth_valid_features.pt            # 驗證集特徵
│   └── nsynth_test_features.pt             # 測試集特徵
│
├── speaker_recognition/        # LibriSpeech 說話人識別
│   ├── config.py              # 配置檔案
│   ├── extract_train-clean-100_features.py        # 訓練集特徵提取
│   ├── extract_testclean_librispeech_features.py  # 測試集特徵提取
│   ├── librispeech_linear_classifier.py           # 單層分類器訓練
│   ├── librispeech_trainclean100_features.pt     # 訓練集特徵
│   ├── librispeech_testclean_features.pt         # 測試集特徵
│   ├── librispeech_linear_classifier.pt          # 單層分類器模型
│   ├── librispeech_three_classifier.pt           # 三層分類器模型
│   ├── README.md              # 說話人識別說明
│   └── README_SpeakerRecognition.md             # 詳細說明
│
├── key_word_spotting/          # Speech Commands 關鍵詞檢測
│   ├── extract_spc_features_audiosep_v1.py       # v1 特徵提取
│   ├── extract_spc_features_audiosep_v2.py       # v2 特徵提取
│   ├── train_spc_fc_classifiers.py               # 分類器訓練
│   ├── spc_features_audiosep_v1.pt               # v1 特徵
│   └── spc_features_audiosep_v2.pt               # v2 特徵
│
├── pretrained_models/          # 預訓練模型
│   └── AudioSep/
│       └── pytorch_model.bin   # AudioSep 預訓練權重
│
└── README.md                   # 本檔案
```

## 🎯 下游任務說明

### 1. ESC-50 環境音分類
- **資料集**: ESC-50 環境音分類資料集
- **任務**: 50個環境音類別分類
- **樣本數**: 2,000個音頻樣本
- **音頻長度**: 5秒
- **挑戰**: 樣本數量少，類別不平衡
- **AudioSep 優勢**: 分離能力有助於提取環境音特徵

### 2. ICBHI 醫學音頻分類
- **資料集**: ICBHI 呼吸音資料集
- **任務**: 正常/異常呼吸音二分類
- **樣本數**: 920個音頻樣本
- **應用**: 醫學診斷輔助
- **挑戰**: 數據量小，類別不平衡
- **AudioSep 優勢**: 分離能力有助於提取呼吸音特徵

### 3. UrbanSound8K 城市聲音分類
- **資料集**: UrbanSound8K 城市聲音資料集
- **任務**: 10個城市聲音類別分類
- **樣本數**: 8,732個音頻樣本
- **應用**: 城市環境監控
- **挑戰**: 音頻長度變化大，背景噪音
- **AudioSep 優勢**: 分離能力有助於去除背景噪音

### 4. NSynth 樂器分類
- **資料集**: NSynth 合成樂器資料集
- **任務**: 10個樂器家族分類
- **樣本數**: 305,979個音頻樣本
- **音頻長度**: 4秒
- **挑戰**: 音頻質量高，需要精細特徵
- **AudioSep 優勢**: 分離能力有助於提取樂器特徵

### 5. LibriSpeech 說話人識別
- **資料集**: LibriSpeech 語音資料集
- **任務**: 248個說話人識別
- **樣本數**: 28,539個音頻樣本
- **應用**: 說話人驗證、語音助手
- **挑戰**: 說話人數量多，需要區分性特徵
- **AudioSep 優勢**: 分離能力有助於提取說話人特徵

### 6. Speech Commands 關鍵詞檢測
- **資料集**: Speech Commands 關鍵詞資料集
- **任務**: 35個關鍵詞檢測 (v1) / 36個關鍵詞檢測 (v2)
- **樣本數**: 105,829個音頻樣本
- **音頻長度**: 1秒
- **應用**: 語音助手、智能家居
- **挑戰**: 音頻短，需要快速識別
- **AudioSep 優勢**: 分離能力有助於提取關鍵詞特徵

## 🚀 快速開始

### 環境設置
```bash
# 創建虛擬環境
python -m venv .venv

# 啟動虛擬環境
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安裝依賴
pip install torch torchaudio
pip install librosa
pip install scikit-learn
pip install numpy pandas
pip install tqdm
```

### 下載預訓練模型
```bash
# 確保 pretrained_models/AudioSep/pytorch_model.bin 存在
# 或從官方來源下載 AudioSep 預訓練模型
```

### 執行所有下游任務評估

#### 1. ESC-50 環境音分類
```bash
cd esc50_feature_fc

# 提取特徵
python extract_esc50_features_audiosep.py

# 訓練分類器
python esc50_linear_classifier.py --epochs 100
```

#### 2. ICBHI 醫學音頻分類
```bash
cd icbhi_feature_fc

# 提取特徵
python extract_icbhi_features_audiosep.py

# 訓練分類器
python icbhi_linear_classifier.py --epochs 100
```

#### 3. UrbanSound8K 城市聲音分類
```bash
cd urbansound8k_feature_fc

# 提取特徵
python extract_urbansound8k_features_audiosep.py

# 訓練分類器
python urbansound8k_linear_classifier.py --epochs 100
```

#### 4. NSynth 樂器分類
```bash
cd nsynth_feature_fc

# 提取特徵
python extract_nsynth_train_features_audiosep.py
python extract_nsynth_valid_features_audiosep.py
python extract_nsynth_test_features_audiosep.py

# 訓練分類器
python train_nsynth_fc_classifiers_valid_test.py
```

#### 5. LibriSpeech 說話人識別
```bash
cd speaker_recognition

# 提取特徵
python extract_train-clean-100_features.py
python extract_testclean_librispeech_features.py

# 訓練分類器
python librispeech_linear_classifier.py
```

#### 6. Speech Commands 關鍵詞檢測
```bash
cd key_word_spotting

# 提取特徵 (選擇 v1 或 v2)
python extract_spc_features_audiosep_v1.py
# 或
python extract_spc_features_audiosep_v2.py

# 訓練分類器
python train_spc_fc_classifiers.py
```

## 📊 模型架構

### AudioSep 特徵提取
```python
def extract_audiosep_features(waveform):
    # 1. 頻譜圖轉換
    mag, cos_in, sin_in = model.wav_to_spectrogram_phase(waveform)
    
    # 2. 批歸一化
    x = model.bn0(mag.transpose(1, 3)).transpose(1, 3)
    
    # 3. 編碼器提取特徵
    x = model.pre_conv(x)
    x = encoder_blocks(x)  # 6個編碼器塊
    
    # 4. 全局平均池化
    pooled = x.mean(dim=[2, 3])  # [B, 384]
    
    return pooled
```

### 分類器架構

#### 單層分類器
```python
class LinearClassifier(nn.Module):
    def __init__(self, in_dim=384, num_classes=None):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)
```

#### 三層分類器
```python
class ThreeLayerClassifier(nn.Module):
    def __init__(self, in_dim=384, num_classes=None, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
```

## 🔧 配置說明

### 統一實驗配置
```python
EXPERIMENT_CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-3,
    'epochs': 100,
    'feature_dim': 384,
    'train_test_split': 0.2,
    'random_seed': 42,
    'device': 'cuda'
}
```

### 特徵標準化
所有任務都使用 `StandardScaler` 進行特徵標準化：
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_np = scaler.fit_transform(features.numpy())
features = torch.tensor(features_np, dtype=torch.float32)
```

## 📈 預期結果

### 性能基準 (示例)
| 任務 | 單層分類器 | 三層分類器 | 最佳方法 |
|------|------------|------------|----------|
| ESC-50 | ~85% | ~88% | 三層 |
| ICBHI | ~75% | ~78% | 三層 |
| UrbanSound8K | ~80% | ~83% | 三層 |
| NSynth | ~90% | ~92% | 三層 |
| Speaker Recognition | ~85% | ~88% | 三層 |
| Speech Commands | ~95% | ~96% | 三層 |

*注意：實際性能可能因硬體配置和數據預處理而有所不同*

## 🔍 實驗分析

### 特徵分析
- **特徵可視化**: 使用 t-SNE 或 UMAP 可視化特徵分佈
- **任務相似性**: 分析不同任務間的特徵相似性
- **消融實驗**: 評估不同組件的貢獻

### 性能分析
- **準確率**: 主要評估指標
- **F1分數**: 用於不平衡數據集
- **混淆矩陣**: 詳細分類分析
- **標準差**: 多次實驗的穩定性

## 🎯 研究貢獻

1. **系統性評估**: 首次全面評估 AudioSep 在6個不同下游任務上的表現
2. **任務覆蓋面廣**: 涵蓋環境音、醫學音頻、城市聲音、樂器、說話人和關鍵詞檢測
3. **實用指南**: 提供基於 AudioSep 的音頻表示學習最佳實踐
4. **開源框架**: 提供完整的實驗代碼和預訓練模型

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

### 貢獻流程
1. Fork 本專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權

本專案採用 MIT 授權。

## 🔗 相關連結

- [AudioSep 論文](https://arxiv.org/abs/2308.03247)
- [AudioSep 官方實現](https://github.com/Audio-AGI/AudioSep)
- [ESC-50 資料集](https://github.com/karolpiczak/ESC-50)
- [ICBHI 資料集](https://bhichallenge.med.auth.gr/)
- [UrbanSound8K 資料集](https://urbansounddataset.weebly.com/)
- [NSynth 資料集](https://magenta.tensorflow.org/datasets/nsynth)
- [LibriSpeech 資料集](https://www.openslr.org/12/)
- [Speech Commands 資料集](https://www.tensorflow.org/datasets/catalog/speech_commands)

## 📞 聯繫方式

如有問題或建議，請通過以下方式聯繫：
- 提交 GitHub Issue
- 發送郵件至：[您的郵箱]

---

**注意**: 本專案僅用於學術研究目的。請確保遵守相關資料集的使用條款和授權協議。 
