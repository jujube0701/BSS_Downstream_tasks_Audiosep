import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from models.audiosep import AudioSep, get_model_class
from config import PRETRAINED_MODEL_PATH

# ====== 參數區 ======
csv_path = 'LibriSpeech/metadata/test-clean.csv'
audio_dir = 'LibriSpeech'
output_path = 'speaker_recognition/librispeech_testclean_features.pt'
batch_size = 1
# ====== END ======

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_dir, transform=None):
        import pandas as pd
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['origin_path'].replace('\\', '/'))
        waveform, sr = torchaudio.load(file_path)
        if self.transform:
            waveform = self.transform(waveform)
        label = int(row['speaker_ID'])
        return waveform, label

def extract_features(audiosep, dataloader, device, save_interval=1, resume_idx=0):
    features, labels = [], []
    for i, (waveforms, label_ids) in enumerate(tqdm(dataloader)):
        global_idx = i + resume_idx
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
        # 每個都存一個 pt，檔名用全局 index
        part_path = os.path.join(os.path.dirname(output_path), f'{os.path.basename(output_path)}_part_{global_idx+1}.pt')
        if os.path.exists(part_path):
            print(f"跳過已存在: {part_path}")
        else:
            torch.save({'features': torch.cat(features, dim=0), 'labels': torch.cat(labels, dim=0)}, part_path)
            print(f"已保存 {global_idx+1} 筆特徵到 {part_path}")
        features, labels = [], []

if __name__ == '__main__':
    device = torch.device('cpu')  # 強制用CPU
    print(f"Device: {device}")
    ResUNet30 = get_model_class('ResUNet30')
    ss_model = ResUNet30(input_channels=1, output_channels=1, condition_size=512)
    audiosep = AudioSep(ss_model=ss_model, query_encoder=None)
    checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location='cpu')
    if "model" in checkpoint:
        audiosep.load_state_dict(checkpoint["model"], strict=False)
    else:
        audiosep.load_state_dict(checkpoint, strict=False)
    audiosep.eval()
    for param in audiosep.parameters():
        param.requires_grad = False
    audiosep = audiosep.to(device)

    dataset = LibriSpeechDataset(csv_path, audio_dir)
    # 自動 resume：偵測已完成的分批檔案
    features_dir = os.path.dirname(output_path) or '.'
    part_files = sorted(glob.glob(os.path.join(features_dir, os.path.basename(output_path) + '_part_*.pt')),
                       key=lambda x: int(x.split('_part_')[-1].split('.pt')[0]))
    resume_idx = 0
    if part_files:
        last_idx = int(part_files[-1].split('_part_')[-1].split('.pt')[0])
        resume_idx = last_idx
        print(f'偵測到已完成 {resume_idx} 筆，將從第 {resume_idx} 筆繼續')
        dataset = Subset(dataset, range(resume_idx, len(dataset)))
    else:
        print('未偵測到分批檔案，將從頭開始')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    extract_features(audiosep, dataloader, device, save_interval=1, resume_idx=resume_idx)
    print('剩餘特徵已全部補齊！') 