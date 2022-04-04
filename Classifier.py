import torch
import torch.nn as nn
from unet import UNet


class Classifier(nn.Module):

  def __init__(self, num_classes, use_pretrained=False, freeze_encoder=False):
    super(Classifier, self).__init__()

    self.unet = UNet(
      in_channels=3,
      out_channels=3,
      n_blocks=5,
      start_filters=32,
      activation='relu',
      normalization='batch',
      conv_mode='same',
      dim=2
    )
    if use_pretrained:
      self.unet.load_state_dict(torch.load("models/unet_65_adam_0.0005.pt"))
      if freeze_encoder:
        for param in self.unet.parameters():
          param.requires_grad = False 

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512 * 14 * 14, num_classes)
    )


  def forward(self, x):
    return self.classifier(self.unet(x, encode_only=True))
