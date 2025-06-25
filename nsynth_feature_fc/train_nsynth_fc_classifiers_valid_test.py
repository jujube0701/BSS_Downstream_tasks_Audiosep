import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 參數
VALID_FEATURE_PATH = 'nsynth_feature_fc/nsynth_valid_features.pt'
VALID_JSON_PATH = 'nsynth/nsynth-valid/examples.json'
TEST_FEATURE_PATH = 'nsynth_feature_fc/nsynth_test_features.pt'
TEST_JSON_PATH = 'nsynth/nsynth-test/examples.json'
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

# 1. 讀取 valid 特徵
print('Loading valid features...')
valid_data = torch.load(VALID_FEATURE_PATH)
valid_features = valid_data['features']
valid_keys = valid_data['keys']

# 2. 讀取 valid label
print('Loading valid labels...')
with open(VALID_JSON_PATH, 'r') as f:
    valid_meta = json.load(f)
valid_labels = []
valid_features_filtered = []
for i, k in enumerate(valid_keys):
    try:
        label = int(valid_meta[k]['instrument_family'])
        if 0 <= label < 10:
            valid_labels.append(label)
            valid_features_filtered.append(valid_features[i])
    except Exception as e:
        continue
valid_features = torch.stack(valid_features_filtered)
valid_labels = torch.tensor(valid_labels)

# 3. 讀取 test 特徵
print('Loading test features...')
test_data = torch.load(TEST_FEATURE_PATH)
test_features = test_data['features']
test_keys = test_data['keys']

# 4. 讀取 test label
print('Loading test labels...')
with open(TEST_JSON_PATH, 'r') as f:
    test_meta = json.load(f)
test_labels = []
test_features_filtered = []
for i, k in enumerate(test_keys):
    try:
        label = int(test_meta[k]['instrument_family'])
        if 0 <= label < 10:
            test_labels.append(label)
            test_features_filtered.append(test_features[i])
    except Exception as e:
        continue
test_features = torch.stack(test_features_filtered)
test_labels = torch.tensor(test_labels)

num_classes = len(set(valid_labels.tolist() + test_labels.tolist()))
print(f'num_classes: {num_classes}, valid: {len(valid_labels)}, test: {len(test_labels)}')

# 5. DataLoader
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = torch.utils.data.DataLoader(SimpleDataset(valid_features, valid_labels), batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(SimpleDataset(test_features, test_labels), batch_size=BATCH_SIZE, shuffle=False)

# 6. 定義模型
class FC1(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

class FC3(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# 7. 訓練與測試

def train_and_test(model, train_loader, test_loader, epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Train'):
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.view(xb.size(0), -1)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        # 測試
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.view(xb.size(0), -1)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        print(f'Epoch {epoch+1}: test acc = {acc:.4f}')
    print(f'Final test acc: {acc:.4f}')
    return acc

# 8. 執行
in_dim = valid_features[0].numel()
print('\n訓練一層全連接分類器...')
fc1 = FC1(in_dim, num_classes)
train_and_test(fc1, train_loader, test_loader, EPOCHS, LEARNING_RATE)

print('\n訓練三層全連接分類器...')
fc3 = FC3(in_dim, num_classes)
train_and_test(fc3, train_loader, test_loader, EPOCHS, LEARNING_RATE) 