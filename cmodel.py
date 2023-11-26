import numpy as np
import os
import glob
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image 
from tqdm import tqdm

import cv2
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset
from torchinfo import summary

# tensor変換
class DBTransform(): 
    #   https://qiita.com/kurilab/items/b69e1be8d0224ae139ad
    def __init__(self, phase="train"):
        self.phase = phase
        self.data_transform = {
            "train": A.Compose([
                A.OneOf([
                    # Blur
                    A.Blur(blur_limit=(3, 7)),
                    # MotionBlur
                    A.MotionBlur(blur_limit=(3, 7)),
                    # GaussianBlur
                    A.GaussianBlur(blur_limit=(3, 7)),
                    # GlassBlur
                    A.GlassBlur(0.7,4,2),
                    # GaussNoise
                    A.GaussNoise(10.0, 50.0),
                    A.GaussNoise(100),
                    # ISONoise
                    # A.ISONoise(1.5, 1.5),
                    A.ISONoise(intensity=(0.5, 1.5),color_shift=(0.1, 0.9),always_apply=True),
                    # MultiplicativeNoise
                    A.MultiplicativeNoise(0.9, 1.1),
                    # Downscale
                    A.Downscale(scale_min=0.5, scale_max=0.9),  # 例として scale_min と scale_max を指定
                    # Flip
                    A.Flip(),
                    # VerticalFlip
                    A.VerticalFlip(),
                    # HorizontalFlip
                    A.HorizontalFlip(),
                    # RandomRotate90
                    A.RandomRotate90(),
                    # RandomScale
                    A.RandomScale(),
                    # Transpose
                    A.Transpose(),
                    # ShiftScaleRotate
                    A.ShiftScaleRotate(),
                    # OpticalDistortion
                    A.OpticalDistortion(),
                    # GridDistortion
                    A.GridDistortion(),
                    # ElasticTransform
                    A.ElasticTransform(),
                    # HueSaturationValue
                    # A.HueSaturationValue(),
                    A.HueSaturationValue(hue_shift_limit=(-20, 20),sat_shift_limit=(-30, 30),val_shift_limit=(-20, 20),p=0.5),
                    # RGBShift
                    A.RGBShift(),
                    # Posterize
                    A.Posterize(),
                    # ChannelDropout
                    A.ChannelDropout(),
                    # ToGray
                    A.ToGray(),
                    # ToSepia
                    A.ToSepia(),
                    # InvertImg
                    A.InvertImg(),
                    # RandomGamma
                    A.RandomGamma(),
                    # RandomBrightnessContrast
                    A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05)),
                    # A.RandomBrightnessContrast(brightness_limit=[-0.05, 0.05], contrast_limit=[-0.05, 0.05]),
                    # CLAHE
                    A.CLAHE(),
                    # Solarize
                    A.Solarize(),
                    # CoarseDropout
                    A.CoarseDropout(),
                    # RandomSnow
                    A.RandomSnow(),
                    # RandomRain
                    A.RandomRain(),
                    # RandomFog
                    A.RandomFog(),
                    # RandomSunFlare
                    A.RandomSunFlare(),
                    # RandomShadow
                    A.RandomShadow(),
                    # FancyPCA
                    A.FancyPCA(),
                    # PadIfNeeded
                    A.PadIfNeeded(),
                ]),
            A.Resize(28,28)]),
            "valid": A.Compose([
                A.OneOf([
                    # GaussianBlur
                    A.GaussianBlur(blur_limit=(3, 7)),
                    # GaussNoise
                    A.GaussNoise(10.0, 50.0),
                    A.GaussNoise(100),
                    # MultiplicativeNoise
                    A.MultiplicativeNoise(0.9, 1.1),
                    # Transpose
                    A.Transpose(),
                    # GridDistortion
                    A.GridDistortion(),
                    # ElasticTransform
                    A.ElasticTransform(),
                    # RGBShift
                    A.RGBShift(),
                    # Posterize
                    A.Posterize(),
                    # ToGray
                    A.ToGray(),
                    # ToSepia
                    A.ToSepia(),
                    # InvertImg
                    A.InvertImg(),
                    # RandomGamma
                    A.RandomGamma(),
                    # Solarize
                    A.Solarize(),
                    # CoarseDropout
                    A.CoarseDropout(),
                    # RandomSnow
                    A.RandomSnow(),
                    # RandomRain
                    A.RandomRain(),
                    # RandomFog
                    A.RandomFog(),
                    # RandomSunFlare
                    A.RandomSunFlare(),
                    # RandomShadow
                    A.RandomShadow(),
                    # FancyPCA
                    A.FancyPCA(),
                    # PadIfNeeded
                    A.PadIfNeeded(),
                ]),
            A.Resize(28,28)]),
        }
    
    def custom_resize(self, img):
        return cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
    
    def __call__(self, img):
        # return self.data_transform[self.phase](image=img)
        # PIL Image から NumPy 配列に変換
        img_np = np.array(img)
        albumentations_transformed = self.data_transform[self.phase](image=img_np)
        # 'image'キーを使用してNumpyイメージを取り出す
        img_np_transformed = albumentations_transformed['image']
        # NumpyイメージをPyTorchのテンソルに変換
        tensor_transform = transforms.ToTensor()
        return {'image': tensor_transform(img_np_transformed)}

# 学習モデル
# チャンネル：[64, 3, 7, 7]
class DBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.fc = nn.Linear(1000,10) 
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        # これを有効にしないと、計算した勾配が毎回異なり、再現性が担保できない。
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# デバイスを選択する。
device = get_device(use_gpu=True)

# 画像のパスのリストを取得する関数
def get_image_path_list(root_path, phase='train'):
    target_path = os.path.join(root_path, phase, '**/*.jpg')
    img_path_list = glob.glob(target_path)
    return img_path_list

root_path = './MachineLearning/db_select/data'
# train_list = get_image_path_list(root_path, phase='train')
# valid_list = get_image_path_list(root_path, phase='valid')
# print(train_list)
# print(valid_list)
trainDir = './MachineLearning/db_select/data/train'
validDir = './MachineLearning/db_select/data/valid'

# ImageFolderは第一引数はディレクトリにする必要がある
train_dataset = torchvision.datasets.ImageFolder(root=trainDir, transform=DBTransform(phase="train"))
valid_dataset = torchvision.datasets.ImageFolder(root=validDir, transform=DBTransform(phase="valid"))

batch_size = 16

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=False)

dataloader_dict = {"train": train_dataloader, "valid": valid_dataloader}  

net = DBNet()
net.cuda()  # GPU対応
# net = net.to(device)

# 交差エントロピー誤差関数
loss_fnc = nn.CrossEntropyLoss()

# 最適化アルゴリズム
learning_rate = 0.1
optimizer = optim.Adam(net.parameters())

# ハイパーパラメータの設定
num_epochs = 50         # 学習を何回、繰り返すか
num_batch = 20         # 1度に、何枚の画像を取出すか

# 損失のログ
record_loss_train = []
record_loss_test = []

# Albumentationsは中身の参照が
# データローダから取得されたバッチが (x, t) の形ではなく、{'image': tensor} の形になっている
# 学習
for i in range(num_epochs):
    net.train()  # 訓練モード
    loss_train = 0
    for j, batch in enumerate(train_dataloader):  # ミニバッチ（x, t）を取り出す
        batch_dict = batch[0]  # バッチ内の辞書を取り出す
        x, t = batch_dict['image'].cuda(), batch[1].cuda()  # GPU対応。'target'はbatch[1]
        y = net(x)
        loss = loss_fnc(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    record_loss_train.append(loss_train)
    
    net.eval()  # 評価モード
    loss_test = 0
    for j, batch in enumerate(valid_dataloader):  # ミニバッチ（x, t）を取り出す
        batch_dict = batch[0]  # バッチ内の辞書を取り出す
        x, t = batch_dict['image'].cuda(), batch[1].cuda()  # GPU対応。'target'はbatch[1]
        y = net(x)
        loss = loss_fnc(y, t)
        loss_test += loss.item()
    loss_test /= j+1
    record_loss_test.append(loss_test)

    if i % 10 == 9:
        print("Epoch:", i, "Loss_Train:", loss_train, "Loss_Test:", loss_test)

correct = 0
total = 0
net.eval()  # 評価モード
for j, batch in enumerate(valid_dataloader):
    batch_dict = batch[0]  # バッチ内の辞書を取り出す
    x, t = batch_dict['image'].cuda(), batch[1].cuda()  # GPU対応。'target'はbatch[1]
    y = net(x)
    correct += (y.argmax(1) == t).sum().item()
    total += len(x)

totalCorrect = correct / total * 100
print("正解率:", str(totalCorrect) + "%")

# 誤差
plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
