import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class SiameseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)

        # Remove the classifier
        self.backbone.fc = nn.Identity()

    def forward_once(self, x):
        return self.backbone(x)

    def forward(self, img1, img2):

        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)

        return embedding1, embedding2