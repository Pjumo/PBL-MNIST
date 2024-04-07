import torch.utils.data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def img_preprocessing(tensor):
    images_raw = torch.squeeze(tensor)
    images_raw = images_raw.numpy()
    # images_pre = images_raw.copy()
    # for image_idx, image_raw in enumerate(images_raw):
    #     image_pre = image_raw.copy()
    #     random_idx = np.random.randint(6, 25, size=2)
    #     if image_raw[random_idx[0]][random_idx[1]] != -1:
    #         for i in range(3):
    #             for j in range(3):
    #                 image_pre[random_idx[0] - i + 1][random_idx[1] - j + 1] = -1
    #     images_pre[image_idx] = image_pre
    # return images_pre
    images_raw = Image.fromarray(images_raw)
    image_pre = transforms.RandomResizedCrop((28, 28), scale=(0.8, 1.2))(images_raw)
    return image_pre


batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 전체 데이터셋을 불러옵니다.
full_dataset = datasets.MNIST('./', train=True, transform=transform, download=True)

# 데이터셋을 훈련, 검증, 테스트 세트로 나눕니다.
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# DataLoader를 생성합니다.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

images, labels = next(iter(train_loader))

image_preprocessing = img_preprocessing(images[0])
image_preprocessing = np.array(image_preprocessing)

image = images[0]
image = torch.squeeze(image)
image = image.numpy()

fig = plt.figure()
sub1 = fig.add_subplot(1, 2, 1)
sub1.imshow(image, 'gray')
sub2 = fig.add_subplot(1, 2, 2)
sub2.imshow(image_preprocessing, 'gray')

plt.show()
