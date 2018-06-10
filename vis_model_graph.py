import torch
import torch.nn as nn
import torch.utils.data
from torchvision.models import resnet101
from tensorboardX import SummaryWriter

# input
dummy_input = torch.autograd.Variable(torch.rand(1, 3, 224, 224))

# model
model = resnet101()

#model.fc = nn.Linear(model.fc.in_features,100)

with SummaryWriter(comment='Net') as w:
    w.add_graph(model, (dummy_input, ))


# >>run this code, the model graph will be saved in runs
# >>turn on the terminal
# >>cd project path
# >>run 'tensorboard --logdir runs'
# >>choose the model graph