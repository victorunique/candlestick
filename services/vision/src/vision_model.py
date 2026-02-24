import torch
import torch.nn as nn

class LightweightCNN(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class VisionPipeline:
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = LightweightCNN().to(self.device)
        
        # Use fp16 for MPS
        if self.device.type == "mps":
            self.model = self.model.half()
            
        self.model.eval()

    def extract_features(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tensor = tensor.to(self.device)
            if self.device.type == "mps":
                tensor = tensor.half()
            # If shape is (1, 64, 1000), make it (1, 1, 64, 1000) for batch and channel
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                tensor = tensor.unsqueeze(1)
            elif tensor.dim() == 3: # e.g. (64, 1000)
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            return self.model(tensor)
