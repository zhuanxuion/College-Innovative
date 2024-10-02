import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from pathlib import Path


def createDeepLabv3(outputchannels=1, exp_directory="CFExp", inherit=False):
    if inherit:
        exp_directory = Path(exp_directory)
        model = torch.load(exp_directory / 'weights.pt')
    else:
        model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT',
                                                        progress=True)
        model.classifier = DeepLabHead(2048, outputchannels)
        # Set the model in training mode
    model.train()
    return  model
