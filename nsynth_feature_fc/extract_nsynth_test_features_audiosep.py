import os
import sys
import json
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../AudioSep-main'))
from models.audiosep import AudioSep, get_model_class
from config import PRETRAINED_MODEL_PATH

# ====== 參數區 ======
json_path = 'nsynth/nsynth-test/examples.json'
audio_dir = 'nsynth/nsynth-test/audio'
output_path = 'nsynth_feature_fc/nsynth_test_features.pt'
batch_size = 1
# ====== END ======

class NSynthTestDataset(Dataset):
    def __init__(self, json_path, audio_dir):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        wav_path = os.path.join(self.audio_dir, key + '.wav')
        waveform, sr = torchaudio.load(wav_path)
        return waveform, key

def extract_features(audiosep, dataloader, device, save_interval=100, resume_idx=0):
    features, keys = [], []
    for i, (waveforms, key_list) in enumerate(tqdm(dataloader)):
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
            pooled = x_center.mean(dim=[2, 3])
            features.append(pooled.cpu())
            keys.extend(key_list)
        del pooled, x, x_center, x6_pool, mag, cos_in, sin_in, waveforms
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 分批保存
        if (global_idx + 1) % save_interval == 0:
            part_path = os.path.join(os.path.dirname(output_path), f'nsynth_test_features_part_{global_idx+1}.pt')
            torch.save({'features': torch.cat(features, dim=0), 'keys': keys}, part_path)
            print(f"已保存 {global_idx+1} 筆特徵到 {part_path}")
            features, keys = [], []
    # 最後一批
    if features:
        part_path = os.path.join(os.path.dirname(output_path), f'nsynth_test_features_part_{global_idx+1}.pt')
        torch.save({'features': torch.cat(features, dim=0), 'keys': keys}, part_path)
        print(f"已保存 {global_idx+1} 筆特徵到 {part_path}")

def merge_feature_parts():
    features_dir = os.path.dirname(output_path) or '.'
    part_files = sorted(glob.glob(os.path.join(features_dir, 'nsynth_test_features_part_*.pt')), 
                       key=lambda x: int(x.split('_part_')[-1].split('.pt')[0]))
    all_features, all_keys = [], []
    for pf in part_files:
        data = torch.load(pf)
        all_features.append(data['features'])
        all_keys.extend(data['keys'])
    features = torch.cat(all_features, dim=0)
    torch.save({'features': features, 'keys': all_keys}, output_path)
    print(f'所有分批特徵已合併並保存到 {output_path}，總共 {features.shape[0]} 筆')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    dataset = NSynthTestDataset(json_path, audio_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 自動 resume：偵測已完成的分批檔案
    features_dir = os.path.dirname(output_path) or '.'
    part_files = sorted(glob.glob(os.path.join(features_dir, 'nsynth_test_features_part_*.pt')),
                       key=lambda x: int(x.split('_part_')[-1].split('.pt')[0]))
    resume_idx = 0
    if part_files:
        last_idx = int(part_files[-1].split('_part_')[-1].split('.pt')[0])
        resume_idx = last_idx
        print(f'偵測到已完成 {resume_idx} 筆，將從第 {resume_idx} 筆繼續')
        dataset = torch.utils.data.Subset(dataset, range(resume_idx, len(dataset)))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        print('未偵測到分批檔案，將從頭開始')

    extract_features(audiosep, dataloader, device)
    merge_feature_parts()
    print(f'特徵已保存到 {output_path}') 