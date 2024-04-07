import torch
import torch.utils.data
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(p=0.25),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(p=0.25),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 64, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = CNN()

num_epochs = 30
batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation((-10, 10)),
    transforms.RandomResizedCrop((28, 28), scale=(0.8, 1.2)),
    transforms.Normalize((0.5,), (0.5,))
])

# 전체 데이터셋을 불러옵니다.
full_dataset = datasets.MNIST('./', train=True, transform=transform, download=True)

# 데이터셋을 훈련, 검증, 테스트 세트로 나눕니다.
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

test_dataset = datasets.MNIST('./', train=False, transform=transform)

# DataLoader를 생성합니다.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda epoch_: 0.95 ** epoch_,
    last_epoch=-1)

best_val_acc = 0.0  # 최고 검증 정확도를 저장하기 위한 변수

for epoch in range(num_epochs):
    model.train()  # 모델을 훈련 모드로 설정
    total_train_acc = 0
    total_train_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        predict = model(images)
        loss = torch.nn.CrossEntropyLoss()(predict, labels.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_acc += (predict.argmax(1) == labels).float().mean().item()
        total_train_loss += loss.item()

    total_train_loss /= len(train_loader)
    total_train_acc /= len(train_loader)

    model.eval()  # 모델을 평가 모드로 설정
    total_val_acc = 0
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            predict = model(images)
            acc = (predict.argmax(1) == labels).float().mean().item()
            loss = torch.nn.CrossEntropyLoss()(predict, labels.long())
            total_val_acc += acc
            total_val_loss += loss.item()

        total_val_loss /= len(val_loader)
        total_val_acc /= len(val_loader)

    print(f'Epoch {epoch + 1} - , Train Acc: {total_train_acc:.4f}, Validation Acc: {total_val_acc:.4f}')

    if total_val_acc > best_val_acc:
        best_val_acc = total_val_acc

    scheduler.step()

model.eval()
total_test_acc = 0
total_test_loss = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        predict = model(images)
        acc = (predict.argmax(1) == labels).float().mean().item()
        loss = torch.nn.CrossEntropyLoss()(predict, labels.long())
        total_test_acc += acc
        total_test_loss += loss.item()

    total_test_loss /= len(test_loader.dataset)
    total_test_acc /= len(test_loader)

print(f'Best Validation Acc: {best_val_acc:.4f}')
print(f'Test Loss: {total_test_loss:.4f}, Test Acc: {total_test_acc:.4f}')

model = model.to("cpu")
torch.cuda.empty_cache()
