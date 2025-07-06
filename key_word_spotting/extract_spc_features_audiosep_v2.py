import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../AudioSep-main'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../speaker_recognition'))

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from models.audiosep import AudioSep, get_model_class
from config import PRETRAINED_MODEL_PATH

# ====== 參數區 ======
data_root = 'Speech commands/v2'
output_path = 'key_word_spotting/spc_features_audiosep_v2.pt'
batch_size = 1
# ====== END ======

# 掃描所有類別
all_classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('_')]
all_classes.sort()
label2idx = {c: i for i, c in enumerate(all_classes)}

# 收集所有 wav 路徑與 label
samples = []
for c in all_classes:
    class_dir = os.path.join(data_root, c)
    for fname in os.listdir(class_dir):
        if fname.endswith('.wav'):
            samples.append((os.path.join(class_dir, fname), label2idx[c], c + '/' + fname))

print(f'共 {len(samples)} 條樣本，{len(all_classes)} 類')

class SPCDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        wav_path, label, key = self.samples[idx]
        waveform, sr = torchaudio.load(wav_path)
        return waveform, label, key

def extract_features(audiosep, dataloader, device):
    features, labels, keys = [], [], []
    for waveforms, label_ids, key_list in tqdm(dataloader):
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
            labels.append(label_ids)
            keys.extend(key_list)
        del pooled, x, x_center, x6_pool, mag, cos_in, sin_in, waveforms
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    features = torch.cat(features, dim=0)
    labels = torch.tensor(labels).view(-1)
    torch.save({'features': features, 'labels': labels, 'keys': keys, 'label2idx': label2idx}, output_path)
    print(f'特徵已保存到 {output_path}')

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

    dataset = SPCDataset(samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    extract_features(audiosep, dataloader, device) 