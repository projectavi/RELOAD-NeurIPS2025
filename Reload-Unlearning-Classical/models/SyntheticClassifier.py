import torch
import torch.nn as nn

class SyntheticClassifier(nn.Module):
    def __init__(self, in_features=2, num_classes=2, **kwargs):
        super(SyntheticClassifier, self).__init__()
        self.in_features = in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

def synthetic_classifier(pretrained=False, progress=True, **kwargs):
    return SyntheticClassifier(in_features=2, **kwargs)