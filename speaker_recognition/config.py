import os

# 獲取項目根目錄
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# AudioSep預訓練模型路徑
PRETRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'pretrained_models', 'AudioSep', 'pytorch_model.bin')

# LibriSpeech 相關路徑
LIBRISPEECH_FEATURES_PATH = os.path.join(PROJECT_ROOT, 'librispeech_features_audiosep.pt')
LIBRISPEECH_SCALER_PATH = os.path.join(PROJECT_ROOT, 'librispeech_scaler.pt')

# 分類器保存路徑
SPEAKER_CLASSIFIER_LINEAR_PATH = os.path.join(PROJECT_ROOT, 'speaker_classifier_linear.pt')
SPEAKER_CLASSIFIER_THREE_LAYER_PATH = os.path.join(PROJECT_ROOT, 'speaker_classifier_three_layer.pt')

# 模型配置
FEATURE_DIM = 512  # AudioSep提取的特徵維度
SAMPLE_RATE = 32000  # 音頻採樣率
MAX_DURATION = 10.0  # 音頻最大長度（秒）

# 訓練配置
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 32 