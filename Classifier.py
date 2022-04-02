import torch
import torch.nn as nn
from unet import UNet


class Classifier(nn.Module):

	def __init__(self, num_classes, use_pretrained=False):
		super(Classifier, self).__init__()

		if not use_pretrained:
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
		# else:
			# self.unet = torch.load("models/unet.pt")

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(512 * 14 * 14, num_classes),
			nn.Softmax(dim=-1)
		)


	def forward(self, x):
		return self.classifier(self.unet(x, encode_only=True))