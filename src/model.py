import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights

def build_model(name: str, num_classes: int, dropout: float = 0.2):
    name = name.lower()
    if name == 'resnet18':
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_feats = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, num_classes))
    elif name == 'efficientnet_b0':
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError('model_name must be resnet18 or efficientnet_b0')
    return m
