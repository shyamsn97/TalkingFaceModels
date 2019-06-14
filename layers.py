import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from utils import *

def conv2d(in_channels, out_channels, bias = True, kernel_size = 1, stride=1, padding=0, 
		   spectral=True, init_zero_weights=False,activation=None,pool=None,norm=None):
	"""
	"""
	layers = []
	if padding > 0:
		layers.append(nn.ReflectionPad2d(padding))
	conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=bias)
	if init_zero_weights:
		conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
	if spectral:
		   conv_layer = nn.utils.spectral_norm(conv_layer)
	layers.append(conv_layer)
	if norm:
		layers.append(norm)
	if activation:
		layers.append(activation)
	if pool:
		layers.append(pool)
	return torch.nn.Sequential(*layers)

def conv1d(in_channels, out_channels, bias = True, kernel_size = 1, stride=1, padding=0, 
		   spectral=True, init_zero_weights=False,activation=None,pool=None,norm=None):
	"""
	"""
	layers = []
	if padding > 0:
		layers.append(nn.ReflectionPad1d(padding))
	conv_layer = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=bias)
	if init_zero_weights:
		conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
	if spectral:
		   conv_layer = nn.utils.spectral_norm(conv_layer)
	layers.append(conv_layer)
	if norm:
		layers.append(norm)
	if activation:
		layers.append(activation)
	if pool:
		layers.append(pool)
	return torch.nn.Sequential(*layers)

class SelfAttn(nn.Module):
	""" 
		Self attention Layer
		taken from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
	"""
	def __init__(self,in_dim):
		super(SelfAttn,self).__init__()
		self.chanel_in = in_dim
		# TODO - test with spectral normalization
		self.query_conv = conv2d(in_channels = in_dim , out_channels = in_dim//8 , 
								 kernel_size = 1, spectral = False)
		self.key_conv = conv2d(in_channels = in_dim , out_channels = in_dim//8 , 
							   kernel_size = 1, spectral = False)
		self.value_conv = conv2d(in_channels = in_dim , out_channels = in_dim , 
								 kernel_size = 1, spectral = False)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax  = nn.Softmax(dim=-1) 
	
	def forward(self,x):
		"""
			inputs :
				x : input feature maps( B X C X W X H)
			returns :
				out : self attention value + input feature 
				attention: B X N X N (N is Width*Height)
		"""
		m_batchsize,C,width ,height = x.size()
		proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
		proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
		energy =  torch.bmm(proj_query,proj_key) # transpose check
		attention = self.softmax(energy) # BX (N) X (N) 
		proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

		out = torch.bmm(proj_value,attention.permute(0,2,1) )
		out = out.view(m_batchsize,C,width,height)
		
		out = self.gamma*out + x
		return out,attention

"""
	Residual Layers
"""
class Res(nn.Module):
	def __init__(self,inp_channel,right_channel):
		super(Res, self).__init__()
		"""
			Residual layer, much like this:
			recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
		"""
		# right side
		convr1 = conv2d(in_channels = inp_channel, out_channels = right_channel, 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=nn.ReLU(),
					   pool = None,norm =None)
		
		convr2 = conv2d(in_channels = right_channel, out_channels = inp_channel, 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = None,norm =None)
		self.right_side = nn.Sequential(*[convr1,convr2])
		
	def forward(self, x):
		o = self.right_side(x)
		return o + x


class ResIn(nn.Module):
	def __init__(self,inp_channel,right_channel):
		super(ResIn, self).__init__()
		"""
			Residual layer, much like this:
			recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
			with instance norm
		"""
		# right side
		convr1 = conv2d(in_channels = inp_channel, out_channels = right_channel, 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=nn.ReLU(),
					   pool = None,norm = nn.InstanceNorm2d(num_features=right_channel,affine=True))
		
		convr2 = conv2d(in_channels = right_channel, out_channels = inp_channel, 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = None,norm = nn.InstanceNorm2d(num_features=right_channel,affine=True))
		self.right_side = nn.Sequential(*[convr1,convr2])
		
	def forward(self, x):
		o = self.right_side(x)
		return o + x


class ResAdaIn(nn.Module):
	def __init__(self,inp_channel,right_channel):
		super(ResAdaIn, self).__init__()
		"""
			Residual layer, much like this:
			recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
			with adaptive instance norm
		"""
		# right side
		self.convr1 = conv2d(in_channels = inp_channel, out_channels = right_channel, 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = None, norm = None)
		# ***adaIN*** #
		self.relu = nn.ReLU()
		self.convr2 = conv2d(in_channels = right_channel, out_channels = inp_channel, 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = None,norm = None)
		# ***adaIN*** #
		
	def forward(self, x, projection):
		right = self.convr2(x)
		right = adaIn(right,projection)
		right = self.relu(right)
		right = self.convr2(right)
		right = adaIn(right,projection)
		return x + right

class ResDown(nn.Module):
	def __init__(self,inp_channel,right_channels,down_scale):
		super(ResDown, self).__init__()
		"""
			ResDown layer from BigGAN
			https://github.com/ajbrock/BigGAN-PyTorch
			# right_channels [0,1] len 2
			# spectral normalization applied, in spots like in ResUp
		"""
		#input img of size N x C x H x W (N is batch size, C is number of channels)
		# right side
		relur1 = nn.ReLU()
		convr1 = conv2d(in_channels = inp_channel, out_channels = right_channels[0], 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=nn.ReLU(),
					   pool = None,norm = None) 
		# N x right_channels[0] x H x W 
		convr2 = conv2d(in_channels = right_channels[0], out_channels = right_channels[1], 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = nn.AvgPool2d(down_scale),norm = None)
		# N x right_channels[1] x H // 2 x W // 2
		self.right_side = nn.Sequential(*[relur1,convr1,convr2])
		#left side
		convl1 = conv2d(in_channels = inp_channel, out_channels = right_channels[1], 
					   bias = True, kernel_size = 1, stride = 1, padding=0, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = nn.AvgPool2d(down_scale),norm = None)
		# N x right_channels[1] x H // 2 x W // 2
		self.left_side = nn.Sequential(*[convl1])
	
	def forward(self,x):
		left = self.left_side(x)
		right = self.right_side(x)
		return left + right

class ResDownIn(nn.Module):
	def __init__(self,inp_channel,right_channels,down_scale):
		super(ResDownIn, self).__init__()
		"""
			ResDown layer from BigGAN
			https://github.com/ajbrock/BigGAN-PyTorch
			# right_channels [0,1] len 2
			# Instance normalization and spectral normalization applied, in spots like in ResUp
		"""
		#input img of size N x C x H x W (N is batch size, C is number of channels)
		# right side
		instance_normr1 = nn.InstanceNorm2d(num_features=inp_channel,affine=True)
		relur1 = nn.ReLU()
		convr1 = conv2d(in_channels = inp_channel, out_channels = right_channels[0], 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=nn.ReLU(),
					   pool = None, norm = nn.InstanceNorm2d(num_features=right_channels[0],affine=True)) 
		# N x right_channels[0] x H x W 
		convr2 = conv2d(in_channels = right_channels[0], out_channels = right_channels[1], 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = nn.AvgPool2d(down_scale), norm = None)
		# N x right_channels[1] x H // 2 x W // 2
		self.right_side = nn.Sequential(*[instance_normr1,relur1,convr1,convr2])
		#left side
		convl1 = conv2d(in_channels = inp_channel, out_channels = right_channels[1], 
					   bias = True, kernel_size = 1, stride = 1, padding=0, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = nn.AvgPool2d(down_scale),norm = nn.InstanceNorm2d(num_features=right_channels[1]))
		# N x right_channels[1] x H // 2 x W // 2
		self.left_side = nn.Sequential(*[convl1])
	
	def forward(self,x):
		left = self.left_side(x)
		right = self.right_side(x)
		return left + right

class ResUpAdaIn(nn.Module):
	def __init__(self,inp_channel,right_channels,up_scale):
		super(ResUpAdaIn, self).__init__()
		"""
			ResUp layer adapted from BigGAN
			https://github.com/ajbrock/BigGAN-PyTorch 
			# right_channels [0,1] len 2
			# Adaptive Instance normalization and spectral normalization applied
		"""
		#input img of size N x C x H x W (N is batch size, C is number of channels)
		# right side
		# ***adIn*** #
		self.relur1 = nn.ReLU()
		self.upsampler1 = nn.UpsamplingNearest2d(scale_factor=up_scale) 
		# N x C x H * 2 x W * 2
		self.convr1 = conv2d(in_channels = inp_channel, out_channels = right_channels[0], 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = None,norm = None) 
		# N x right_channels[0] x H * 2 x W * 2
		# ***adIn*** #
		self.relur2 = nn.ReLU()
		self.convr2 = conv2d(in_channels = right_channels[0], out_channels = right_channels[1], 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = None,norm = None) 
		# N x right_channels[1] x H * 2 x W * 2 
		# left size
		self.upsamplel1 = nn.UpsamplingNearest2d(scale_factor=up_scale)
		# N x C x H * 2 x W * 2
		convl1 = conv2d(in_channels = inp_channel, out_channels = right_channels[1], 
					   bias = True, kernel_size = 1, stride = 1, padding=0, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = None,norm = None)
		# N x right_channels[1] x H * 2 x W * 2
	
	def forward(self,x, projection):
		# projection_moments are affine coefficients for adaIn, generated by the meta embeddings and projection matrix
		#left side
		left = self.upsamplel1(x)
		left = self.convl1(left)
		#right side
		right = adaIn(x,projection) # can be inefficient
		right = self.relur1(right)
		right = self.upsampler1(right)
		right = self.convr1(right)
		right = adaIn(right,projection)
		right = self.relur2(right)
		right = self.convr2(right)
		return left + right

class ResUpAdaInV2(nn.Module):
	def __init__(self, inp_channel,right_channels,projection_dim,up_scale):
		super(ResUpAdaInV2, self).__init__()
		"""
			ResUp layer adapted from BigGAN
			https://github.com/ajbrock/BigGAN-PyTorch 
			# right_channels [0,1] len 2
			# Adaptive Instance normalization and spectral normalization applied, with linear
		"""
		#input img of size N x C x H x W (N is batch size, C is number of channels)
		# right side
		# linears are implemented as convolutions
		self.linear1 = conv1d(in_channels = projection_dim , out_channels = inp_channel, 
					   bias = False, kernel_size = 1, stride = 1, padding = 0, 
					   spectral = True, init_zero_weights = False, activation = None,
					   pool = None,norm = None)
		# ***adIn*** #
		self.relur1 = nn.ReLU()
		self.upsampler1 = nn.UpsamplingNearest2d(scale_factor=up_scale) 
		# N x C x H * 2 x W * 2
		self.convr1 = conv2d(in_channels = inp_channel, out_channels = right_channels[0], 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = None,norm = None) 
		# N x right_channels[0] x H * 2 x W * 2
		self.linear2 = conv1d(in_channels = projection_dim , out_channels = right_channels[0], 
					   bias = False, kernel_size = 1, stride = 1, padding = 0, 
					   spectral = True, init_zero_weights = False, activation = None,
					   pool = None,norm = None)
		# ***adIn*** #
		self.relur2 = nn.ReLU()
		self.convr2 = conv2d(in_channels = right_channels[0], out_channels = right_channels[1], 
					   bias = True, kernel_size = 3, stride = 1, padding=1, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = None,norm = None) 
		# N x right_channels[1] x H * 2 x W * 2 
		# left size
		self.upsamplel1 = nn.UpsamplingNearest2d(scale_factor=up_scale)
		# N x C x H * 2 x W * 2
		convl1 = conv2d(in_channels = inp_channel, out_channels = right_channels[1], 
					   bias = True, kernel_size = 1, stride = 1, padding=0, 
					   spectral = True, init_zero_weights = False,activation=None,
					   pool = None,norm = None)
		# N x right_channels[1] x H * 2 x W * 2
	
	def forward(self,x, projection):
		# projection_moments are affine coefficients for adaIn, generated by the meta embeddings and projection matrix
		#left side
		left = self.upsamplel1(x)
		left = self.convl1(left)
		#right side
		linearproj1 = self.linear1(projection)
		right = adaIn(x,linearproj1) # can be inefficient
		right = self.relur1(right)
		right = self.upsampler1(right)
		right = self.convr1(right)
		linearproj2 = self.linear2(projection)
		right = adaIn(right,linearproj2)
		right = self.relur2(right)
		right = self.convr2(right)
		return left + right


