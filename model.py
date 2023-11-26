import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import albumentations as A

# 学習済みモデルと同じ並びリストとして設定が必要
classes_ja = ["ベジット", "セル", "フリーザ", "ゴジータ", "孫悟飯", "孫悟空", "孫悟天", "魔人ブウ", "トランクス", "ベジータ"]
classes_en = ["Begetto", "Cell", "Freeza", "Gogeta", "Gohan", "Gokuu", "Goten", "MajinBuu", "Trunks", "Vegeta"]
n_class = len(classes_ja)
img_size = 28

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

net = DBNet()

# 評価モードに設定して学習処理を動かないようにする
net.eval()

# 保存された重みを読み込む
pretrained_dict = torch.load("./MachineLearning/db_select/dbmodel_cnn.pth", map_location=torch.device("cpu"))

# モデルに読み込んだ重みをロード
net.load_state_dict(pretrained_dict)
    
def predict(img):    
    # 画像変換定義
    transform = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_LINEAR)])

    # PIL ImageをNumPy配列に変換
    img = np.array(img)

    # グレースケール画像か確認
    if len(img.shape) == 2:
        # 新しい変換
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # グレースケールからRGBに変換
    elif len(img.shape) == 3 and img.shape[2] == 1:
        # 新しい変換
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # グレースケールからRGBに変換

    # Albumentationsを使った変換
    img = transform(image=img)['image']

    # NumPyイメージをPyTorchのテンソルに変換
    transform = transforms.ToTensor()
    x = transform(img)

    # バッチ次元を追加
    x = x.unsqueeze(0)

    # 予測
    y = net(x)

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
