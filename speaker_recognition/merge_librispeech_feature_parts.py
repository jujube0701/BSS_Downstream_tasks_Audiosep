import torch
import glob
import os

output_path = 'speaker_recognition/librispeech_devclean_features.pt'
part_files = sorted(glob.glob('speaker_recognition/librispeech_devclean_features.pt_part_*.pt'),
                    key=lambda x: int(x.split('_part_')[-1].split('.pt')[0]))

all_features, all_labels = [], []
for pf in part_files:
    print(f'正在合併: {pf}')
    data = torch.load(pf)
    all_features.append(data['features'])
    all_labels.append(data['labels'])
features = torch.cat(all_features, dim=0)
labels = torch.cat(all_labels, dim=0)
torch.save({'features': features, 'labels': labels}, output_path)
print(f'所有分批特徵已合併並保存到 {output_path}，總共 {features.shape[0]} 筆')
print('features shape:', features.shape)
print('labels shape:', labels.shape) 