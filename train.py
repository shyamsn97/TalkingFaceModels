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

class Composer:
	def __init__(self,projection_dims,embedding_dims,reshaped_sequences,landmark_sequences,landmark_sequence_count,num_sequences,k=8,checkpoint=None):
		self.projection_dims = projection_dims
		self.embedding_dims = embedding_dims
		self.num_sequences = num_sequences
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
																lr=5.0 * 10**-5)
		self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
														lr=2.0 * 10**-4)

		if checkpoint:
			self.generator.load_state_dict(checkpoint['generator'])
			self.embedder.load_state_dict(checkpoint['embedder'])
			self.discriminator.load_state_dict(checkpoint['discriminator'])
			self.generator_embedder_optimizer.load_state_dict(checkpoint['generator_embedder_optimizer'])			
			self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.loss = Loss()
		self.args = {
			"vgg19_layers": [1,6,11,20,29],
			"vggface_layers": [1,6,11,18,25],
			"vgg19_weight": 10.0**(-2), 
			"vggface_weight": 10.0**(-3)
		}

	def make_models(self,projection_dims,embedding_dims,num_sequences):
		generator = GeneratorV3(projection_dims)
		print("Successfully created Generator")
		embedder = Embedder(embedding_dims)
		print("Successfully created Embedder")
		discriminator = Discriminator(num_sequences,projection_dims)
		print("Successfully created Discriminator")

		return generator, embedder, discriminator

	def toDict(self,epoch):
		args = {}
		args["epoch"] = epoch
		args["projection_dims"] = self.projection_dims
		args["embedding_dims"] = self.embedding_dims
		args["num_sequences"] = self.num_sequences
		args["generator"] = self.generator.state_dict()
		args["embedder"] = self.embedder.state_dict()
		args["discriminator"] = self.discriminator.state_dict()
		args["generator_embedder_optimizer"] = self.generator_embedder_optimizer.state_dict()
		args["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()
		return args

	def imgsToDevice(self,target,sampled_vids):
		target[0] = target[0].float()
		target[1] = target[1].float()
		for i in range(len(sampled_vids)):
			sampled_vids[i][0] = sampled_vids[i][0].float()
			sampled_vids[i][1] = sampled_vids[i][1].float()
		return target, sampled_vids

	def reset_grad(self):
		self.discriminator_optimizer.zero_grad()
		self.generator_embedder_optimizer.zero_grad()

	def metaTrain(self,epochs,save=True):
		now = str(datetime.datetime.now()) + "_{}_{}_{}_epochs".format(self.projection_dims,self.embedding_dims,epochs)
		saver = Saver(now)
		saver.makeDir()
		print("Training for {} epochs".format(epochs))
		bar = tqdm(np.arange(epochs))
		generator_losses = [None]
		discriminator_losses = [None]
		self.generator.train()
		self.embedder.train()
		self.discriminator.train()
		for epoch in bar:
			bar.set_description("Generator + Embedder loss: {}, Discriminator loss: {}".format(generator_losses[-1],discriminator_losses[-1]))
			gen_loss = 0.0
			disc_loss = 0.0
			for index, data in enumerate(self.dataloader):
				target, sampled_vids = data
				target, sampled_vids = self.imgsToDevice(target,sampled_vids)
				x, y = target
				avg_embedding = self.embedder.average_embeddings(sampled_vids,w_out_grad=False)
				x_hat = self.generator(y,avg_embedding)

				# Discriminator Loss
				self.reset_grad()
				real_realism, real_activations, real_spliced = self.discriminator(x,y,index)
				fake_realism, fake_activations, fake_spliced = self.discriminator(x_hat,y,index)
				discriminator_loss = self.loss.discriminatorLoss(fake_realism,real_realism)
				discriminator_loss.backward()
				self.discriminator_optimizer.step()
				disc_loss += discriminator_loss.item()
				
				# Update Discrim and Generator Loss
				self.reset_grad()
				avg_embedding = self.embedder.average_embeddings(sampled_vids)
				x_hat = self.generator(y,avg_embedding)
				real_realism, real_activations, real_spliced = self.discriminator(x,y,index)
				fake_realism, fake_activations, fake_spliced = self.discriminator(x_hat,y,index)
				generator_embedder_loss = self.loss.generatorLoss(x,x_hat,fake_realism,
																real_activations,fake_activations,avg_embedding,real_spliced,self.args)
				generator_embedder_loss.backward()
				self.generator_embedder_optimizer.step()
				gen_loss += generator_embedder_loss.item()

			if save:
				saver.saveCheckpoint(epoch,self.toDict(epoch))

			generator_losses.append(gen_loss)
			discriminator_losses.append(disc_loss)
		saver.saveCheckpoint("final",self.toDict("final"))




	def metaTrainV2(self,epochs,save=True):
		now = str(datetime.datetime.now()) + "_{}_{}_{}_epochs".format(self.projection_dims,self.embedding_dims,epochs)
		saver = Saver(now)
		saver.makeDir()
		print("Training for {} epochs".format(epochs))
		bar = tqdm(np.arange(epochs))
		generator_losses = [-1]
		discriminator_losses = [-1]
		self.generator.train()
		self.embedder.train()
		self.discriminator.train()
		for epoch in bar:
			bar.set_description("Generator + Embedder loss: {}, Discriminator loss: {}".format(generator_losses[-1],discriminator_losses[-1]))
			gen_loss = 0.0
			disc_loss = 0.0
			for index, data in enumerate(self.dataloader):
				target, sampled_vids = data
				target, sampled_vids = self.imgsToDevice(target,sampled_vids)
				x, y = target
				avg_embedding = self.embedder.average_embeddings(sampled_vids)
				x_hat = self.generator(y,avg_embedding)

				real_realism, real_activations, real_spliced = self.discriminator(x,y,index)
				fake_realism, fake_activations, fake_spliced = self.discriminator(x_hat,y,index)

				# Discriminator Loss
				self.generator_embedder_optimizer.zero_grad()
				self.discriminator_optimizer.zero_grad()
				generator_embedder_loss = self.loss.generatorLoss(x,x_hat,fake_realism,
																real_activations,fake_activations,avg_embedding,real_spliced,self.args)
				discriminator_loss = self.loss.discriminatorLoss(fake_realism,real_realism)
				loss = generator_embedder_loss + discriminator_loss
				gen_loss += generator_embedder_loss.item()
				disc_loss += discriminator_loss.item()
				loss.backward(retain_graph=True)
				self.generator_embedder_optimizer.step()
				self.discriminator_optimizer.step()
				
				# Generator Loss
				x_hat = self.generator(y,avg_embedding).detach()
				real_realism, real_activations, real_spliced = self.discriminator(x,y,index)
				fake_realism, fake_activations, fake_spliced = self.discriminator(x_hat,y,index)
				self.discriminator_optimizer.zero_grad()
				discriminator_loss = self.loss.discriminatorLoss(fake_realism,real_realism)
				discriminator_loss.backward()
				self.discriminator_optimizer.step()

			if save:
				saver.saveCheckpoint(epoch,self.toDict(epoch))

			generator_losses.append(gen_loss)
			discriminator_losses.append(disc_loss)
		saver.saveCheckpoint("final",self.toDict("final"))


	def metaTrainV3(self,epochs,save=True):
		now = str(datetime.datetime.now()) + "_{}_{}_{}_epochs".format(self.projection_dims,self.embedding_dims,epochs)
		saver = Saver(now)
		saver.makeDir()
		print("Training for {} epochs".format(epochs))
		bar = tqdm(np.arange(epochs))
		generator_losses = [-1]
		discriminator_losses = [-1]
		self.generator.train()
		self.embedder.train()
		self.discriminator.train()
		for epoch in bar:
			bar.set_description("Generator + Embedder loss: {}, Discriminator loss: {}".format(generator_losses[-1],discriminator_losses[-1]))
			gen_loss = 0.0
			disc_loss = 0.0
			for index, data in enumerate(self.dataloader):
				self.generator_embedder_optimizer.zero_grad()
				self.discriminator_optimizer.zero_grad()
				target, sampled_vids = data
				target, sampled_vids = self.imgsToDevice(target,sampled_vids)
				x, y = target
				avg_embedding = self.embedder.average_embeddings(sampled_vids,w_out_grad=False)
				x_hat = self.generator(y,avg_embedding)
				real_realism, real_activations, real_spliced = self.discriminator(x,y,index)
				fake_realism, fake_activations, fake_spliced = self.discriminator(x_hat,y,index)

				self.generator_embedder_optimizer.zero_grad()
				generator_embedder_loss = self.loss.generatorLoss(x,x_hat,fake_realism,
																real_activations,fake_activations,avg_embedding,real_spliced,self.args)
				gen_loss += generator_embedder_loss.item()
				print("GEN LOSS: {}".format(generator_embedder_loss.item()))
				generator_embedder_loss.backward(retain_graph=True)
				self.generator_embedder_optimizer.step()

				# Discriminator Loss
				self.discriminator_optimizer.zero_grad()
				# Compute again, just in case gradients reset
				discriminator_loss = self.loss.discriminatorLoss(fake_realism,real_realism)
				disc_loss += discriminator_loss.item()
				print("DISC LOSS: {}".format(discriminator_loss.item()))
				discriminator_loss.backward()
				self.discriminator_optimizer.step()

			if save:
				saver.saveCheckpoint(epoch,self.toDict(epoch))

			generator_losses.append(gen_loss)
			discriminator_losses.append(disc_loss)
		saver.saveCheckpoint("final",self.toDict("final"))
