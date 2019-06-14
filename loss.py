import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from vgg import *

class Loss:
	def __init__(self,args={}):
		self.args = args
		self.vgg19 = VGG19()
		self.vggface = VGGFace()
		print("Successfully loaded pretrained VGG19 and VGGFace")
	def lCNT(self, x, x_hat, vgg19_layers=[1,6,11,20,29], vggface_layers=[1,6,11,18,25], vgg19_weight=10**(-2), vggface_weight=10**(-3)):
		"""
			perceptual loss between VGG19 and VGGFace
		"""
		vgg19_real, vgg19_real_activations = self.vgg19(x,vgg19_layers)
		vgg19_fake, vgg19_fake_activations = self.vgg19(x_hat,vggface_layers)
		
		vgg19_loss = 0
		for i in range(len(vgg19_layers)):
			vgg19_loss += F.l1_loss(vgg19_real_activations[i],vgg19_fake_activations[i])

		vggface_real, vggface_real_activations = self.vggface(x,vggface_layers)
		vggface_fake, vggface_fake_activations = self.vggface(x_hat,vggface_layers)

		vggface_loss = 0
		for i in range(len(vggface_layers)):
			vggface_loss += F.l1_loss(vggface_real_activations[i],vggface_fake_activations[i])

		return vgg19_weight*vgg19_loss + vggface_weight*vggface_loss

	def lADV(self,fake_realism_score,D_real_activations,D_fake_activations):
		l_fm = self.lFM(D_real_activations,D_fake_activations)
		return (-1*fake_realism_score) + l_fm

	def lFM(self,D_real_activations,D_fake_activations):
		loss = 0
		for i in range(len(D_real_activations)):
			loss += 10*(F.l1_loss(D_real_activations[i],D_fake_activations[i]))
		return loss
	
	def lMCH(self, embedding, splice):
		return F.l1_loss(embedding, splice)

	def discriminatorLoss(self, fake_realism_score, real_realism_score):
		return torch.max(torch.zeros(1),1.0 + fake_realism_score) + torch.max(torch.zeros(1),1.0 - real_realism_score)
	
	def generatorLoss(self,x,x_hat,fake_realism_score,D_real_activations,
						D_fake_activations,embedding,splice,args):
		vgg19_layers = args["vgg19_layers"]
		vggface_layers = args["vggface_layers"]
		vgg19_weight = args["vgg19_weight"]
		vggface_weight = args["vggface_weight"]

		lcnt = self.lCNT(x,x_hat,vgg19_layers,vggface_layers,vgg19_weight,vggface_weight)
		ladv = self.lADV(fake_realism_score,D_real_activations,D_fake_activations)
		lmch = self.lMCH(embedding,splice)
		return lcnt + ladv + lmch
