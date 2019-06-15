from layers import *
from models import *
from data_processing import *
from loss import *
from vgg import *
from Dataset import *
from utils import *

import numpy as np
import torch
import os
import datetime
from tqdm import tqdm

class MetaLearningStage:
	def __init__(self,projection_dims,embedding_dims,reshaped_sequences,landmark_sequences,landmark_sequence_count,num_sequences,k=8,args={}):
		self.dataset = MetaDataset(reshaped_frame_sequences=reshaped_sequences,
									landmark_frame_sequences=landmark_sequences,num_videos=landmark_sequence_count,k=k)
		self.dataloader = makeDataloader(self.dataset)
		print("Successfully created Dataloader")
		self.generator = GeneratorV3(projection_dims)
		print("Successfully created Generator")
		self.embedder = Embedder(embedding_dims)
		print("Successfully created Embedder")
		self.discriminator = Discriminator(num_sequences,projection_dims)
		print("Successfully created Discriminator")

		self.generator_embedder_optimizer = torch.optim.Adam(list(self.generator.parameters())+list(self.embedder.parameters()), 
																lr=1e-3, weight_decay=1e-6)
		self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
														lr=1e-3, weight_decay=1e-6)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.loss = Loss()
		self.args = {
			"vgg19_layers": [1,6,11,20,29],
			"vggface_layers": [1,6,11,18,25],
			"vgg19_weight": 10**(-2), 
			"vggface_weight": 10**(-3)

		}

	# def imgsToDevice(self,target,sampled_vids):
	# 	target[0] = target[0].to(self.device)
	# 	target[1] = target[1].to(self.device)
	# 	for i in range(len(sampled_vids)):
	# 		sampled_vids[i][0] = sampled_vids[i][0].to(self.device)
	# 		sampled_vids[i][1] = sampled_vids[i][1].to(self.device)
	# 	return target, sampled_vids
	
	def imgsToDevice(self,target,sampled_vids):
		target[0] = target[0].float()
		target[1] = target[1].float()
		for i in range(len(sampled_vids)):
			sampled_vids[i][0] = sampled_vids[i][0].float()
			sampled_vids[i][1] = sampled_vids[i][1].float()
		return target, sampled_vids

	def train(self,epochs,save=True):
		print("Training for {} epochs".format(epochs))
		bar = tqdm(np.arange(epochs))
		generator_losses = [-1]
		discriminator_losses = [-1]
		self.generator.train()
		self.embedder.train()
		self.discriminator.train()
		for i in bar:
			bar.set_description("Generator + Embedder loss: {}, Discriminator loss: {}".format(generator_losses[-1],discriminator_losses[-1]))
			gen_loss = 0.0
			disc_loss = 0.0
			for index, data in enumerate(self.dataloader):
				target, sampled_vids = data
				target, sampled_vids = self.imgsToDevice(target,sampled_vids)
				x = target[0]
				y = target[1]



				avg_embedding = self.embedder.average_embeddings(sampled_vids).detach()
				x_hat = self.generator(y,avg_embedding).detach()
				# Discriminator Loss
				self.discriminator_optimizer.zero_grad()
				# Compute again, just in case gradients reset
				real_realism, real_activations, real_spliced = self.discriminator(x,y,index)
				fake_realism, fake_activations, fake_spliced = self.discriminator(x_hat,y,index)
				discriminator_loss = self.loss.discriminatorLoss(fake_realism,real_realism)
				disc_loss += discriminator_loss.item()
				discriminator_loss.backward()
				self.discriminator_optimizer.step()
				
				# Generator Loss
				self.generator_embedder_optimizer.zero_grad()
				avg_embedding = self.embedder.average_embeddings(sampled_vids)
				x_hat = self.generator(y,avg_embedding)
				real_realism, real_activations, real_spliced = self.discriminator(x,y,index)
				fake_realism, fake_activations, fake_spliced = self.discriminator(x_hat,y,index)
				generator_embedder_loss = self.loss.generatorLoss(x,x_hat,fake_realism,
																real_activations,fake_activations,avg_embedding,real_spliced,self.args)
				gen_loss += generator_embedder_loss.item()
				generator_embedder_loss.backward()
				self.generator_embedder_optimizer.step()
			generator_losses.append(gen_loss)
			discriminator_losses.append(disc_loss)
