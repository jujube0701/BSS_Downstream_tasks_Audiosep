import os

# 獲取項目根目錄
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 數據文件路徑
ESC50_FEATURES_PATH = os.path.join(PROJECT_ROOT, 'esc50_features_audiosep.pt')

# 模型保存路徑
LINEAR_CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, 'esc50_linear_classifier.pt')
THREE_CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, 'esc50_three_classifier.pt')

# ESC-50 數據集路徑
ESC50_AUDIO_DIR = os.path.join(PROJECT_ROOT, 'ESC-50-master', 'audio')
ESC50_META_PATH = os.path.join(PROJECT_ROOT, 'ESC-50-master', 'meta', 'esc50.csv')

# 預訓練模型路徑
PRETRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'pretrained_models', 'AudioSep', 'pytorch_model.bin')

# LibriSpeech 相關路徑
LIBRISPEECH_FEATURES_PATH = os.path.join(PROJECT_ROOT, 'librispeech_features_audiosep.pt')
SPEAKER_CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, 'speaker_classifier_librispeech.pt')
LIBRISPEECH_SCALER_PATH = os.path.join(PROJECT_ROOT, 'librispeech_scaler.pt') 