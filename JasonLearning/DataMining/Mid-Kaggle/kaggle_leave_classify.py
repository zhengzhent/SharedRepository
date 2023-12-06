import os
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, if_valid=False):
        super().__init__()
        self.data = csv_file
        self.transform = transform
        self.if_valid = if_valid
        self.root_dir = root_dir

    def __getitem__(self, index):
        if self.if_valid:
            img_path = os.path.join(self.root_dir, self.data.iloc[index, 0])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image
        else:
            img_path, label = self.data.iloc[index]
            img_path = os.path.join(self.root_dir, self.data.iloc[index, 0])
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label

    def __len__(self):
        return len(self.data)


def transform(train=True):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if train:
        return train_transform
    else:
        return test_transform


def dataloader(path_train, path_valid):
    info_train = pd.read_csv(path_train)
    info_valid = pd.read_csv(path_valid)
    root_dir = 'kaggle_data/classify-leaves'

    #   build dict
    label_classes = info_train.iloc[:, 1].unique()
    len_label = len(label_classes)
    class_to_num = dict(zip(label_classes, range(len_label)))
    num_to_class = dict(zip(range(len_label), label_classes))

    #   build dataset
    info_train.iloc[:, 1] = info_train.iloc[:, 1].map(class_to_num)
    full_set = MyDataset(info_train, root_dir=root_dir, transform=transform(train=True), if_valid=False)
    valid_set = MyDataset(info_valid, root_dir=root_dir, transform=transform(train=False), if_valid=True)

    #   split train and test
    train_size = int(0.8 * len(full_set))
    test_size = len(full_set) - train_size
    train_set, test_set = torch.utils.data.random_split(full_set, [train_size, test_size])

    #   dataloader
    full_loader = DataLoader(full_set, batch_size=150, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=150, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=150, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=150, shuffle=False)

    return class_to_num, num_to_class, train_loader, test_loader, valid_loader, full_loader


class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, strides=1, cov1x1=False):
        super().__init__()
        #   transform width and height
        self.cov1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        #   remain width and height
        self.cov2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        #   residual cov
        if cov1x1:
            self.cov3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.cov3 = None

        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.cov1(x)))
        y = self.bn2(self.cov2(y))
        if self.cov3:
            x = self.cov3(x)
        y += x
        return self.relu(y)


def resnet_block(input_channels, output_channels, num_residual, first_block=False):
    blk = []
    for i in range(num_residual):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, output_channels, strides=2, cov1x1=True))
        else:
            blk.append(Residual(output_channels, output_channels))

    return blk


def resnet(num_class):
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, num_class))

    return net


# x = torch.rand(size=(1, 3, 224, 224))
# net = resnet(100)
# for layer in net:
#     x = layer(x)
#     print(layer.__class__.__name__, 'out shape:\t', x.shape)


def calculate_acc(model, dataloader, device):
    correct_num = 0
    total = 0
    with torch.no_grad:
        for data in dataloader:
            i, j = data
            i, j = i.to(device), j.to(device)
            i = model(i)
            _, i_index = torch.max(i.data, 1)
            total += i.size(0)
            correct_num += (i_index == j).sum().item()

    return correct_num / total


#   参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_train = 'kaggle_data/classify-leaves/train.csv'
path_valid = 'kaggle_data/classify-leaves/test.csv'
class_to_num, num_to_class, train_loader, test_loader, valid_loader, full_loader = dataloader(path_train, path_valid)

epochs = 50
model = resnet(176)
model.to(device)

loss = nn.CrossEntropyLoss()
epochs_loss = []
iter_loss = []

#   训练
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for i in range(epochs):
    iter_loss = []
    for code, data in enumerate(train_loader):
        path, y = data
        path, y = path.to(device), y.to(device)
        opt.zero_grad()
        y_hat = model(path)
        ls = loss(y_hat, y)
        ls.backward()
        opt.step()
        iter_loss.append(ls.item())

        if code == 122:
            print(f'accuracy: {calculate_acc(model, train_loader, device)}:f')

    print(f'epoch:{i + 1}, loss:{sum(iter_loss) / len(iter_loss)}')
    epochs_loss.append(sum(iter_loss) / len(iter_loss))

plt.figure()
plt.plot(range(epochs), epochs_loss)
plt.show()

#   验证
model.load_state_dict(torch.load('bca.pth'))
model.eval()

with torch.no_grad():
    total_num = 0
    corr_num = 0
    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        y_hat = model(image)
        _, pre_index = torch.max(y_hat, 1)
        corr_num += (pre_index == label).sum().item()
        total_num += len(label)

    print(corr_num / total_num)

#   测试
model.load_state_dict(torch.load('leaves_parameters.pth'))
model.eval()

with torch.no_grad():
    result = []
    last = []
    for image in valid_loader:
        image = image.to(device)
        y_hat = model(image)
        _, index = torch.max(y_hat, 1)
        result.extend(index.detach().cpu().numpy())

    for i in result:
        last.append(num_to_class[i])

    valid_data = pd.read_csv(path_valid)
    valid_data['label'] = pd.Series(last)
    valid_data.to_csv('submission1.csv', index=None)
