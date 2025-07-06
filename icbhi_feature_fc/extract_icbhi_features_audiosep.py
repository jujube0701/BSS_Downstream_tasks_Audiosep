import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio
import numpy as np
import glob
import sys

# AudioSep 相關
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../AudioSep-main'))
from models.audiosep import AudioSep, get_model_class

ICBHI_AUDIO_DIR = 'ICBHI_final_database'
ICBHI_FEATURES_PATH = 'ICBHI_feature_fc/icbhi_features_audiosep.pt'
PRETRAINED_MODEL_PATH = 'pretrained_models/AudioSep/pytorch_model.bin'

class ICBHIDataset(Dataset):
    def __init__(self, audio_dir, transform=None):
        self.wav_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
        self.transform = transform
        self.failed_files = []

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        file_path = self.wav_files[idx]
        filename = os.path.basename(file_path)
        txt_path = file_path[:-4] + '.txt'
        label = 0  # 預設正常
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt = f.read().lower()
                if ('crackle' in txt) or ('wheeze' in txt):
                    label = 1  # 異常
        try:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                import torchaudio.transforms as T
                resampler = T.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
            if waveform.shape[-1] < 1024:
                pad_len = 1024 - waveform.shape[-1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        except Exception as e:
            # print(f"檔案讀取失敗: {file_path}, error: {e}")
            self.failed_files.append(file_path)
            return torch.zeros(1, 16000), -1
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

def extract_features(model, dataloader, device, output_path=None, resume_idx=0):
    for i, (waveforms, label_ids) in enumerate(tqdm(dataloader)):
        global_idx = i + resume_idx
        if label_ids.item() == -1:
            continue  # 跳過讀取失敗的檔案
        waveforms = waveforms.to(device)
        try:
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
                # 直接存單筆
                part_path = os.path.join(os.path.dirname(output_path), f'icbhi_features_single_{global_idx+1}.pt')
                torch.save({'features': pooled.cpu(), 'labels': label_ids.cpu()}, part_path)
                # print(f"已保存第 {global_idx+1} 筆特徵到 {part_path}")
            del pooled, x, x_center, x6_pool, mag, cos_in, sin_in, waveforms
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            # print(f"特徵提取失敗: {global_idx+1}, error: {e}")
            continue

def merge_feature_parts(output_path):
    features_dir = os.path.dirname(output_path)
    part_files = sorted(glob.glob(os.path.join(features_dir, 'icbhi_features_single_*.pt')), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    all_features, all_labels = [], []
    for pf in part_files:
        data = torch.load(pf)
        feature = data['features'].unsqueeze(0) if data['features'].dim() == 1 else data['features']
        label = data['labels'].unsqueeze(0) if data['labels'].dim() == 0 else data['labels']
        all_features.append(feature)
        all_labels.append(label)
    if all_features:
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        valid_idx = labels != -1
        features = features[valid_idx]
        labels = labels[valid_idx]
        torch.save({'features': features, 'labels': labels}, output_path)
        print(f'所有單筆特徵已合併並保存到 {output_path}，總共 {features.shape[0]} 筆')
    else:
        print('沒有可用的特徵檔案可合併！')

if __name__ == '__main__':
    batch_size = 1
    
    # 檢查GPU可用性
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("GPU不可用，使用CPU")
    
    print(f"Device: {device}")

    # 檢查已處理的文件數量
    features_dir = os.path.dirname(ICBHI_FEATURES_PATH)
    existing_files = glob.glob(os.path.join(features_dir, 'icbhi_features_single_*.pt'))
    if existing_files:
        # 找到最大的文件編號
        max_file_num = max([int(f.split('_')[-1].split('.')[0]) for f in existing_files])
        resume_idx = max_file_num  # 從下一個文件開始
        print(f"發現已處理到第 {max_file_num} 個文件，將從第 {resume_idx + 1} 個文件開始繼續處理")
    else:
        resume_idx = 0
        print("沒有發現已處理的文件，將從頭開始處理")

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

    dataset = ICBHIDataset(ICBHI_AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 如果resume_idx > 0，需要跳過已處理的文件
    if resume_idx > 0:
        # 創建一個新的dataloader，跳過前resume_idx個文件
        dataset = ICBHIDataset(ICBHI_AUDIO_DIR)
        # 手動跳過已處理的文件
        dataset.wav_files = dataset.wav_files[resume_idx:]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"跳過前 {resume_idx} 個文件，從第 {resume_idx + 1} 個文件開始處理")

    extract_features(audiosep, dataloader, device, output_path=ICBHI_FEATURES_PATH, resume_idx=resume_idx)
    merge_feature_parts(ICBHI_FEATURES_PATH)
    print(f'特徵已保存到 {ICBHI_FEATURES_PATH}') 