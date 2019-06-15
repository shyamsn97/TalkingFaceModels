import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
from torchvision.models import vgg

class VGG19(nn.Module):
	"""
		VGG19 Pretrained
	"""
	def __init__(self):
		super(VGG19, self).__init__()
		cuda = torch.cuda.is_available()
		device = torch.device("cuda" if cuda else "cpu")
		self.model = vgg.vgg19(pretrained=True).features.to(device).eval()
		self.layers = list(self.model.children())
		
	def forward(self, x, target_layers):
		activations = []
		for i, layer in enumerate(self.layers):
			x = layer(x)
			if i in target_layers:
				activations.append(x)
		return x, activations


class VGGFace(nn.Module):
	"""
		VGGFace Pretrained
		Adapted from from https://github.com/prlz77/vgg-face.pytorch
	"""
	def __init__(self):
		super(VGGFace, self).__init__()

		self.layers = []
		self.layers.append(nn.Conv2d(3, 64, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		
		self.layers.append(nn.Conv2d(64, 64, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.MaxPool2d(2))
		
		self.layers.append(nn.Conv2d(64, 128, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		
		self.layers.append(nn.Conv2d(128, 128, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.MaxPool2d(2))

		self.layers.append(nn.Conv2d(128, 256, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		
		self.layers.append(nn.Conv2d(256, 256, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		
		self.layers.append(nn.Conv2d(256, 256, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.MaxPool2d(2))
		
		self.layers.append(nn.Conv2d(256, 512, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())

		self.layers.append(nn.Conv2d(512, 512, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		
		self.layers.append(nn.Conv2d(512, 512, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.MaxPool2d(2))
		
		self.layers.append(nn.Conv2d(512, 512, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		
		self.layers.append(nn.Conv2d(512, 512, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		
		self.layers.append(nn.Conv2d(512, 512, 3, stride=1, padding=1))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.MaxPool2d(2))
		
		self.fc6 = nn.Linear(512 * 7 * 7, 4096)
		self.fc7 = nn.Linear(4096, 4096)
		self.fc8 = nn.Linear(4096, 2622)
		self.load_weights()


	def load_weights(self, path="/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/pretrained_vggface/VGG_FACE.t7"):
		model = torchfile.load(path)
		counter = 1
		block = 1
		count = 0
		for i, layer in enumerate(model.modules):
			if layer.weight is not None and i < len(self.layers):
				self.layers[i].weight.data[...] = torch.tensor(layer.weight).view_as(self.layers[i].weight)[...]
				self.layers[i].bias.data[...] = torch.tensor(layer.bias).view_as(self.layers[i].bias)[...]
				
	def forward(self, x, target_layers):
		activations = []
		for i, layer in enumerate(self.layers):
			x = layer(x)
			if i in target_layers:
				activations.append(x)
		return x, activations
