import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(p=0.25)
        )
        self.model2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(p=0.25),
        )
        self.model3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(p=0.25),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.fc(x)
        return x


# def img_show(image, cnt, label):
#     plt.title(f'failure {label}')
#     plt.imshow(image, 'gray')
#     plt.savefig(f'./failure/fail_{cnt}')


def img_preprocessing(tensor):
    images_raw = torch.squeeze(tensor)
    images_raw = images_raw.numpy()
    images_pre = images_raw.copy()
    for image_idx, image_raw in enumerate(images_raw):
        image_pre = image_raw.copy()
        random_idx = np.random.randint(6, 25, size=2)
        if image_raw[random_idx[0]][random_idx[1]] != -1:
            for i in range(3):
                for j in range(3):
                    image_pre[random_idx[0] - i + 1][random_idx[1] - j + 1] = -1
        images_pre[image_idx] = image_pre
    return images_pre


model = CNN()

num_epochs = 50
batch_size = 32

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation((-10, 10)),
    transforms.RandomResizedCrop((28, 28), scale=(0.8, 1.2)),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 전체 데이터셋을 불러옵니다.
train_dataset = datasets.MNIST('./', train=True, transform=transform_train, download=True)
val_dataset = datasets.MNIST('./', train=True, transform=transform_test, download=True)

test_dataset = datasets.MNIST('./', train=False, transform=transform_test, download=True)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(0.2 * num_train))
train_idx, val_idx = indices[split:], indices[:split]
np.random.shuffle(indices)

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# DataLoader를 생성합니다.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda epoch_: 0.95 ** epoch_,
    last_epoch=-1)

best_val_acc = 0.0  # 최고 검증 정확도를 저장하기 위한 변수
best_val_loss = float('Inf')

for epoch in range(num_epochs):
    model.train()  # 모델을 훈련 모드로 설정
    total_train_acc = 0
    total_train_loss = 0
    for images_, labels in train_loader:
        if np.random.randint(0, 1) != 0:
            images = images_.to(device)
            labels = labels.to(device)

            predict = model(images)
            loss = torch.nn.NLLLoss()(predict, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_acc += (predict.argmax(1) == labels).float().mean().item()
            total_train_loss += loss.item()
        else:
            img_preprocess = img_preprocessing(images_)
            img_preprocess = torch.tensor(img_preprocess).reshape(32, 1, 28, 28)

            images = img_preprocess.to(device)
            labels = labels.to(device)

            predict = model(images)
            loss = torch.nn.NLLLoss()(predict, labels.long())

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
            loss = torch.nn.NLLLoss()(predict, labels.long())
            total_val_acc += acc
            total_val_loss += loss.item()

        total_val_loss /= len(val_loader)
        total_val_acc /= len(val_loader)

    print(f'Epoch {epoch + 1} - , Train Acc: {total_train_acc:.4f}, Validation Acc: {total_val_acc:.4f}')

    # if total_val_acc > best_val_acc:
    #     best_val_acc = total_val_acc
    #     torch.save(model.state_dict(), 'best_model.pt')
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        torch.save(model.state_dict(), 'best_model.pt')

    scheduler.step()

model.load_state_dict(torch.load('best_model.pt'))
model.eval()
total_test_acc = 0
total_test_loss = 0
fail_cnt = 0
with torch.no_grad():
    for cnt, data in enumerate(test_loader):
        images = data[0].to(device)
        labels = data[1].to(device)

        predict = model(images)
        # for idx in range(labels.size(dim=-1)):
        #     if predict.argmax(1)[idx] != labels[idx]:
        #         fail_cnt += 1
        #         img = torch.squeeze(data[0][idx])
        #         img = img.numpy()
        #         img_show(img, fail_cnt, data[1][idx])
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
