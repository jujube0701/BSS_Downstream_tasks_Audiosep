import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../AudioSep-main'))
from models.audiosep import AudioSep, get_model_class
from config import ESC50_META_PATH, ESC50_AUDIO_DIR, ESC50_FEATURES_PATH, PRETRAINED_MODEL_PATH
# from models.clap_encoder import CLAP_Encoder  # 不再需要

class ESC50Dataset(Dataset):
    def __init__(self, csv_path, audio_dir, transform=None):
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        waveform, sr = torchaudio.load(file_path)
        if self.transform:
            waveform = self.transform(waveform)
        label = row['target']
        label_text = row['category']  # ESC-50有category欄位
        return waveform, label, label_text

def extract_features(audiosep, dataloader, device, save_interval=100, resume_idx=0):
    features, labels = [], []
    for i, (waveforms, label_ids, label_texts) in enumerate(tqdm(dataloader)):
        global_idx = i + resume_idx
        if global_idx < resume_idx:
            continue
        waveforms = waveforms.to(device)
        with torch.no_grad():
            mag, cos_in, sin_in = audiosep.ss_model.base.wav_to_spectrogram_phase(waveforms)
            x = mag
            x = x.transpose(1, 3)
            x = audiosep.ss_model.base.bn0(x)
            x = x.transpose(1, 3)
            x = F.pad(x, pad=(0, 0, 0, 0))
            x = x[..., 0 : x.shape[-1] - 1]
            x = audiosep.ss_model.base.pre_conv(x)
            _, _ = audiosep.ss_model.base.encoder_block1(x, None)
            _, _ = audiosep.ss_model.base.encoder_block2(_, None)
            _, _ = audiosep.ss_model.base.encoder_block3(_, None)
            _, _ = audiosep.ss_model.base.encoder_block4(_, None)
            _, _ = audiosep.ss_model.base.encoder_block5(_, None)
            x6_pool, _ = audiosep.ss_model.base.encoder_block6(_, None)
            x_center, _ = audiosep.ss_model.base.conv_block7a(x6_pool, None)
            pooled = x_center.mean(dim=[2, 3])  # [B, 384]
            features.append(pooled.cpu())
            labels.append(label_ids.cpu())
        del pooled, x, x_center, x6_pool, mag, cos_in, sin_in, waveforms
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 分批保存
        if (global_idx + 1) % save_interval == 0:
            part_path = os.path.join(os.path.dirname(ESC50_FEATURES_PATH), f'esc50_features_part_{global_idx+1}.pt')
            torch.save({'features': torch.cat(features, dim=0), 'labels': torch.cat(labels, dim=0)}, part_path)
            print(f"已保存 {global_idx+1} 筆特徵到 {part_path}")
            features, labels = [], []
    # 最後一批
    if features:
        part_path = os.path.join(os.path.dirname(ESC50_FEATURES_PATH), f'esc50_features_part_{global_idx+1}.pt')
        torch.save({'features': torch.cat(features, dim=0), 'labels': torch.cat(labels, dim=0)}, part_path)
        print(f"已保存 {global_idx+1} 筆特徵到 {part_path}")

def merge_feature_parts():
    features_dir = os.path.dirname(ESC50_FEATURES_PATH)
    part_files = sorted(glob.glob(os.path.join(features_dir, 'esc50_features_part_*.pt')), 
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
    all_features, all_labels = [], []
    for pf in part_files:
        data = torch.load(pf)
        all_features.append(data['features'])
        all_labels.append(data['labels'])
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({'features': features, 'labels': labels}, ESC50_FEATURES_PATH)
    print(f'所有分批特徵已合併並保存到 {ESC50_FEATURES_PATH}，總共 {features.shape[0]} 筆')

if __name__ == '__main__':
    # 修改路徑參數
    ESC50_META_PATH = 'ESC-50-master/meta/esc50.csv'
    ESC50_AUDIO_DIR = 'ESC-50-master/audio'
    ESC50_FEATURES_PATH = 'esc50_feature_fc/esc50_features_audiosep.pt'
    PRETRAINED_MODEL_PATH = 'pretrained_models/AudioSep/pytorch_model.bin'

    # 使用配置文件中的路徑
    csv_path = ESC50_META_PATH
    audio_dir = ESC50_AUDIO_DIR
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GPU 診斷資訊
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 不載入 CLAP Encoder
    # 只載入 ResUNet30
    ResUNet30 = get_model_class('ResUNet30')
    ss_model = ResUNet30(input_channels=1, output_channels=1, condition_size=512)  # condition_size仍需給定
    # 構建 AudioSep，不傳 query_encoder
    audiosep = AudioSep(ss_model=ss_model, query_encoder=None)
    
    # 使用動態路徑載入預訓練模型
    checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location='cpu')
    print("ckpt keys:", checkpoint.keys())
    if "model" in checkpoint:
        print("ckpt['model'] keys:", checkpoint["model"].keys())
        audiosep.load_state_dict(checkpoint["model"], strict=False)
    else:
        print("ckpt (no 'model') keys:", checkpoint.keys())
        audiosep.load_state_dict(checkpoint, strict=False)
    audiosep.eval()
    for param in audiosep.parameters():
        param.requires_grad = False

    # 將模型移到 GPU
    audiosep = audiosep.to(device)
    print(f"Model device: {next(audiosep.parameters()).device}")

    dataset = ESC50Dataset(csv_path, audio_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 自動偵測已完成的分批檔案，resume
    features_dir = os.path.dirname(ESC50_FEATURES_PATH)
    part_files = sorted(glob.glob(os.path.join(features_dir, 'esc50_features_part_*.pt')), 
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
    resume_idx = 0
    if part_files:
        last_idx = int(part_files[-1].split('_')[-1].split('.')[0])
        resume_idx = last_idx
        print(f'偵測到已完成 {resume_idx} 筆，將從第 {resume_idx} 筆繼續')
        # 跳過已完成的部分
        dataset = torch.utils.data.Subset(dataset, range(resume_idx, len(dataset)))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    extract_features(audiosep, dataloader, device, save_interval=100, resume_idx=resume_idx)
    merge_feature_parts()
    print(f'特徵已保存到 {ESC50_FEATURES_PATH}') 