# models/swin_classifier.py
import torch
import torch.nn as nn
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

class SwinPureClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SwinPureClassifier, self).__init__()
        # 加载适合 384x384 高分辨率输入的预训练 Swin Transformer V2 Base
        weights = Swin_V2_B_Weights.DEFAULT
        self.swin = swin_v2_b(weights=weights)
        # 获取 Swin 最后一层全连接层的输入特征维度
        num_features = self.swin.head.in_features
        # 分类头替换
        self.swin.head = nn.Linear(num_features, num_classes)

    def forward(self, images):
        return self.swin(images)