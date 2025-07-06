import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = 'urbansound8k_feature_fc/urbansound8k_features_audiosep.pt'
LINEAR_CLASSIFIER_PATH = 'urbansound8k_feature_fc/urbansound8k_linear_classifier.pt'
THREE_CLASSIFIER_PATH = 'urbansound8k_feature_fc/urbansound8k_three_classifier.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 載入特徵和標籤
features_labels = torch.load(FEATURES_PATH)
features = features_labels['features']
labels = features_labels['labels']
# 過濾掉 label = -1 的資料
valid_idx = labels != -1
features = features[valid_idx]
labels = labels[valid_idx]

print("features shape:", features.shape)
print("labels shape:", labels.shape)
print("labels min/max:", labels.min().item(), labels.max().item())
print("labels sample:", labels[:20])

# 2. 標準化
scaler = StandardScaler()
features_np = scaler.fit_transform(features.numpy())
features = torch.tensor(features_np, dtype=torch.float32)

# 3. 切分資料
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
    def __init__(self, in_dim=384, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

class ThreeLayerClassifier(nn.Module):
    def __init__(self, in_dim=384, num_classes=10, hidden_dim=256):
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

    num_classes = len(torch.unique(labels))
    in_dim = features.shape[1]

    # 一層全連接層
    model_linear = LinearClassifier(in_dim=in_dim, num_classes=num_classes).to(device)
    optimizer_linear = optim.Adam(model_linear.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 三層全連接層
    model_three = ThreeLayerClassifier(in_dim=in_dim, num_classes=num_classes).to(device)
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
            best_model_state = model_linear.state_dict().copy()
    torch.save(best_model_state, LINEAR_CLASSIFIER_PATH)
    print('一層全連接層分類器已保存到 urbansound8k_linear_classifier.pt')
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
    print('三層全連接層分類器已保存到 urbansound8k_three_classifier.pt')
    print(f'最佳測試集準確率：{best_test_acc:.4f}') 