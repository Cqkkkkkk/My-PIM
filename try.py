import torch
import torch.nn as nn

class WSS(nn.Module):
    def __init__(self, in_channel, num_classes, num_selects) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channel, num_classes)
        self.num_selects = num_selects
    
    def forward(self, x):
        # [B, HxW, C] = x.shape
        # return class_prediction, selectd_features
        h = self.fc(x)
        logits = torch.softmax(h, dim=-1)
        _, ids = torch.sort(logits, dim=-1, descending=True)
        selection = ids[:, :self.num_selects]
        # returns selected features of [B, num_selects, num_classes]
        return logits, torch.gather(x, 1, selection)


data = torch.rand((1, 16, 8))
model = WSS(in_channel=8, num_classes=6, num_selects=4)
logits, features = model(data)

print(features.shape)
