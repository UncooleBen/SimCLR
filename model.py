import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            # 修改第一个卷积层的参数: kernel_size: 7->3, stride: 2->1, padding: 3->1
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

            # 将非线性层和非max_pool层加入编码器f
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)

        # encoder
        # 将list转换为sequential
        self.f = nn.Sequential(*self.f)

        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        # 转换为一维张量 feature: 1 * 2048, out: 1 * 128
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        # 将输出的特征及projection的输出归一化 feature: 2048 * 1, out: 128 * 1
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# # test
# m = Model()
# m.eval()
# x = torch.randn(1, 3, 224, 224)
# x = m(x)
