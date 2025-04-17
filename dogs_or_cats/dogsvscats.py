import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import sigmoid

transform = transforms.Compose([
    transforms.Resize((128, 128)),   # 統一圖片尺寸
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # RGB各通道標準化
])

train_dataset = datasets.ImageFolder(root='./train', transform=transform)
val_dataset = datasets.ImageFolder(root='./val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),   # 輸入通道3(RGB)，輸出16，卷積核3x3
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # 將尺寸減半
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),     # 輸出圖像大小會是 16x16
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)                 # 二分類
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
model = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.BCEWithLogitsLoss()           # 適合二分類
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {running_loss / len(train_loader):.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = sigmoid(outputs) >= 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Validation Accuracy: {100 * correct / total:.2f}%")