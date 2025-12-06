import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchCNN(nn.Module):
    def __init__(self, embedding_dim=384, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(embedding_dim, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x
    
class PatchNN(nn.Module):
    def __init__(self, patch_dim, H, W, num_classes=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(patch_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
        
class GradCAM:
    def __init__(self, model, target_layer): # Target is probably the last layer
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x) 

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        loss = logits[:, class_idx]
        loss.backward()

        grads = self.gradients 
        activations = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.squeeze(0).squeeze(0)

