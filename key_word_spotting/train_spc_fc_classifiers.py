import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 參數 - 選擇要訓練的版本
VERSION = 'v1'  # 改這裡：'v1' 或 'v2'
FEATURE_PATH = f'key_word_spotting/spc_features_audiosep_{VERSION}.pt'
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

print(f'訓練 Speech Commands {VERSION.upper()} 分類器...')

# 1. 讀取特徵
print('Loading features...')
data = torch.load(FEATURE_PATH)
features = data['features']
labels = data['labels']
label2idx = data['label2idx']
num_classes = len(label2idx)
print(f'num_classes: {num_classes}, total: {len(labels)}')

# 2. train/test 劃分
idx = list(range(len(labels)))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
train_features, train_labels = features[train_idx], labels[train_idx]
test_features, test_labels = features[test_idx], labels[test_idx]

# 3. DataLoader
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = torch.utils.data.DataLoader(SimpleDataset(train_features, train_labels), batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(SimpleDataset(test_features, test_labels), batch_size=BATCH_SIZE, shuffle=False)

# 4. 定義模型
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

# 5. 訓練與測試

def train_and_eval(model, train_loader, test_loader, epochs, lr):
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
    test_acc = correct / total
    print(f'Final test acc: {test_acc:.4f}')
    return test_acc

# 6. 執行
in_dim = train_features[0].numel()
print('\n訓練一層全連接分類器...')
fc1 = FC1(in_dim, num_classes)
train_and_eval(fc1, train_loader, test_loader, EPOCHS, LEARNING_RATE)

print('\n訓練三層全連接分類器...')
fc3 = FC3(in_dim, num_classes)
train_and_eval(fc3, train_loader, test_loader, EPOCHS, LEARNING_RATE) 