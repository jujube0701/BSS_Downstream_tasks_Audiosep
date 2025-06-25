import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 直接定義所有用到的路徑
ESC50_FEATURES_PATH = 'esc50_feature_fc/esc50_features_audiosep.pt'
LINEAR_CLASSIFIER_PATH = 'esc50_feature_fc/esc50_linear_classifier.pt'
THREE_CLASSIFIER_PATH = 'esc50_feature_fc/esc50_three_classifier.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 載入特徵和標籤
features_labels = torch.load(ESC50_FEATURES_PATH)
features = features_labels['features']  # [N, 384]
labels = features_labels['labels']      # [N]

print("features shape:", features.shape)
print("labels shape:", labels.shape)
print("labels min/max:", labels.min().item(), labels.max().item())
print("labels sample:", labels[:20])

# 2. 標準化
scaler = StandardScaler()
features_np = scaler.fit_transform(features.numpy())
features = torch.tensor(features_np, dtype=torch.float32)

# 3. 切分資料（stratify）
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# 4. 轉成 TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 5. 定義分類器
class LinearClassifier(nn.Module):
    def __init__(self, in_dim=384, num_classes=50):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

class ThreeLayerClassifier(nn.Module):
    def __init__(self, in_dim=384, num_classes=50, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# 6. 訓練與驗證

def train(model, loader, optimizer, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # 一層全連接層
    model_linear = LinearClassifier().to(device)
    optimizer_linear = optim.Adam(model_linear.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 三層全連接層
    model_three = ThreeLayerClassifier().to(device)
    optimizer_three = optim.Adam(model_three.parameters(), lr=1e-3)

    print('=== 一層全連接層訓練 ===')
    best_model_state = None
    best_test_acc = 0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train(model_linear, train_loader, optimizer_linear, criterion)
        test_loss, test_acc = evaluate(model_linear, test_loader, criterion)
        print(f'[Linear] Epoch {epoch:3d}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model_linear.state_dict().copy()  # 保存最佳狀態

    # 保存最佳模型
    torch.save(best_model_state, LINEAR_CLASSIFIER_PATH)
    print('一層全連接層分類器已保存到 esc50_linear_classifier.pt')
    print(f'最佳測試集準確率：{best_test_acc:.4f}')

    print('\n=== 三層全連接層訓練 ===')
    best_test_acc = 0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train(model_three, train_loader, optimizer_three, criterion)
        test_loss, test_acc = evaluate(model_three, test_loader, criterion)
        print(f'[ThreeLayer] Epoch {epoch:3d}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    torch.save(model_three.state_dict(), THREE_CLASSIFIER_PATH)
    print('三層全連接層分類器已保存到 esc50_three_classifier.pt')
    print(f'最佳測試集準確率：{best_test_acc:.4f}')

   