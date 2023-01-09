import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 가져올 모델과 같은 구조 // same structure with model to load.
model = models.resnet18()
# print(model)
model.fc = nn.Linear(in_features=412, out_features={'num_classes'}, bias=True)

model.load_state_dict(torch.load({'model.pt'}, map_location=device))
