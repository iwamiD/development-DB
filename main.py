import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

#実行ファイルのパスに合わせる
os.chdir(os.path.dirname(os.path.abspath(__file__)))

classes_ja = ["孫悟空", "孫悟飯", "孫悟天", "ベジータ", "トランクス", "魔人ブウ", "セル", "フリーザ", "ゴジータ", "ベジット"]
classes_en = ["Gokuu", "Gohan", "Goten", "Vegeta", "Trunks", "MajinBuu", "Cell", "Freeza", "Gogeta", "Begetto"]
n_class = len(classes_ja)
img_size = 28

# 画像認識のモデル
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv２ = nn.Conv2d(8, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64*4*4, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn2(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

net = Net()

# 訓練済みパラメータの読み込みと設定
net.load_state_dict(torch.load(
    "./AIML_Pytorch/dev_app/model_cnn.pth", map_location=torch.device("cpu")
    ))
    
def predict(img):
    # モデルへの入力
    img = img.convert("L")  # モノクロに変換
    img = img.resize((img_size, img_size))  # サイズを変換
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.0), (1.0))
                                    ])
    img = transform(img)
    x = img.reshape(1, 1, img_size, img_size)
    
    # 予測
    net.eval()
    y = net(x)

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]