import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchCNN(nn.Module):
    """
    CNN for patch token embeddings.
    Input: (batch, T, H, W, embedding_dim)
    Output: logits for binary classification
    """
    def __init__(self, embedding_dim=384, num_classes=2):
        super().__init__()
        # reduce embedding dimension
        self.conv1 = nn.Conv2d(embedding_dim, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # x: (batch, T, H, W, D)
        batch, T, H, W, D = x.shape
        x = x.view(batch*T, H, W, D).permute(0, 3, 1, 2)  # (batch*T, D, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(batch, T, -1)  # (batch, T, 32)
        x = x.mean(dim=1)                     # temporal mean pooling
        x = self.fc(x)
        return x
