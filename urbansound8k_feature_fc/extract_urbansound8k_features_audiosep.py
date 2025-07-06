import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio
import numpy as np
import sys

# AudioSep 相關
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../AudioSep-main'))
from models.audiosep import AudioSep, get_model_class

URBANSOUND8K_META_PATH = 'UrbanSound8K/metadata/UrbanSound8K.csv'
URBANSOUND8K_AUDIO_DIR = 'UrbanSound8K/audio'
URBANSOUND8K_FEATURES_PATH = 'urbansound8k_feature_fc/urbansound8k_features_audiosep.pt'
PRETRAINED_MODEL_PATH = 'pretrained_models/AudioSep/pytorch_model.bin'

class UrbanSound8KDataset(Dataset):
    def __init__(self, csv_path, audio_dir, transform=None):
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        fold = row['fold']
        file_path = os.path.join(self.audio_dir, f'fold{fold}', row['slice_file_name'])
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # 轉單聲道
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        # 新增：過濾太短的音檔
        if waveform.shape[-1] < 1024:
            return torch.zeros(1, 16000), -1
        if self.transform:
            waveform = self.transform(waveform)
        label = int(row['classID'])
        return waveform, label  # shape [1, L]

def extract_features(model, dataloader, device, save_interval=100):
    features, labels = [], []
    for i, (waveforms, label_ids) in enumerate(tqdm(dataloader)):
        waveforms = waveforms.to(device)
        with torch.no_grad():
            mag, cos_in, sin_in = model.ss_model.base.wav_to_spectrogram_phase(waveforms)
            x = mag
            x = x.transpose(1, 3)
            x = model.ss_model.base.bn0(x)
            x = x.transpose(1, 3)
            x = torch.nn.functional.pad(x, pad=(0, 0, 0, 0))
            x = x[..., 0 : x.shape[-1] - 1]
            x = model.ss_model.base.pre_conv(x)
            _, _ = model.ss_model.base.encoder_block1(x, None)
            _, _ = model.ss_model.base.encoder_block2(_, None)
            _, _ = model.ss_model.base.encoder_block3(_, None)
            _, _ = model.ss_model.base.encoder_block4(_, None)
            _, _ = model.ss_model.base.encoder_block5(_, None)
            x6_pool, _ = model.ss_model.base.encoder_block6(_, None)
            x_center, _ = model.ss_model.base.conv_block7a(x6_pool, None)
            pooled = x_center.mean(dim=[2, 3])
            features.append(pooled.cpu())
            labels.append(label_ids.cpu())
        if (i + 1) % save_interval == 0:
            part_path = f'urbansound8k_feature_fc/urbansound8k_features_audiosep_part_{i+1}.pt'
            torch.save({'features': torch.cat(features, dim=0), 'labels': torch.cat(labels, dim=0)}, part_path)
            print(f"已保存 {i+1} 筆特徵到 {part_path}")
            features, labels = [], []
    if features:
        part_path = f'urbansound8k_feature_fc/urbansound8k_features_audiosep_part_{i+1}.pt'
        torch.save({'features': torch.cat(features, dim=0), 'labels': torch.cat(labels, dim=0)}, part_path)
        print(f"已保存 {i+1} 筆特徵到 {part_path}")

def merge_feature_parts():
    import glob
    part_files = sorted(glob.glob('urbansound8k_feature_fc/urbansound8k_features_audiosep_part_*.pt'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    all_features, all_labels = [], []
    for pf in part_files:
        data = torch.load(pf)
        all_features.append(data['features'])
        all_labels.append(data['labels'])
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({'features': features, 'labels': labels}, URBANSOUND8K_FEATURES_PATH)
    print(f'所有分批特徵已合併並保存到 {URBANSOUND8K_FEATURES_PATH}，總共 {features.shape[0]} 筆')

if __name__ == '__main__':
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ResUNet30 = get_model_class('ResUNet30')
    ss_model = ResUNet30(input_channels=1, output_channels=1, condition_size=512)
    audiosep = AudioSep(ss_model=ss_model, query_encoder=None)
    checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
    if "model" in checkpoint:
        audiosep.load_state_dict(checkpoint["model"], strict=False)
    else:
        audiosep.load_state_dict(checkpoint, strict=False)
    audiosep.eval()
    for param in audiosep.parameters():
        param.requires_grad = False
    audiosep = audiosep.to(device)

    dataset = UrbanSound8KDataset(URBANSOUND8K_META_PATH, URBANSOUND8K_AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    extract_features(audiosep, dataloader, device)
    merge_feature_parts()
    print(f'特徵已保存到 {URBANSOUND8K_FEATURES_PATH}') 