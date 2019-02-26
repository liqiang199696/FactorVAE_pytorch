import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class Discriminator(nn.Module):
	def __init__(self, args):
		super(Discriminator, self).__init__()
		self.z_dim = args.z_dim
		
		self.fc1 = nn.Linear(self.z_dim, 1000)
		self.fc2 = nn.Linear(1000, 1000)
		self.fc3 = nn.Linear(1000, 1000)
		self.fc4 = nn.Linear(1000, 1000)
		self.fc5 = nn.Linear(1000, 1000)
		self.fc6 = nn.Linear(1000, 2)
		self.leakyrelu = nn.LeakyReLU(0.2, True)

	def forward(self, z):
		# print('Discriminator z: ',z.size())
		z = self.leakyrelu(self.fc1(z))
		# print('Discriminator z: ',z.size())
		z = self.leakyrelu(self.fc2(z))
		z = self.leakyrelu(self.fc3(z))
		z = self.leakyrelu(self.fc4(z))
		z = self.leakyrelu(self.fc5(z))
		z = self.fc6(z)
		return z




# MNIST encoder and decoder
class Encoder_MNIST(nn.Module):
	def __init__(self,args):
		super(Encoder_MNIST,self).__init__()
		self.z_dim = args.z_dim

		self.conv1 = nn.Conv2d(1,32,3)
		self.conv2 = nn.Conv2d(32,64,3)

		self.fc1 = nn.Linear(64*5*5,256)
		self.fc2 = nn.Linear(256,120)
		self.fc3 = nn.Linear(120,84)
		self.fc4 = nn.Linear(84,self.z_dim*2)

		self.pool = nn.MaxPool2d(2,2)

	def forward(self,x):
		#print('MNIST encoder x: ',x.size())
		x = self.pool(F.relu(self.conv1(x)))
		#print('MNIST encoder x: ',x.size())
		x = self.pool(F.relu(self.conv2(x)))
		#print('MNIST encoder x: ',x.size())
		x = x.view(-1,self.num_flat_features(x))
		#print('MNIST encoder x: ',x.size())
		x = F.relu(self.fc1(x))
		#print('MNIST encoder x: ',x.size())
		x = F.relu(self.fc2(x))
		#print('MNIST encoder x: ',x.size())
		x = F.relu(self.fc3(x))
		#print('MNIST encoder x: ',x.size())
		x = self.fc4(x)
		#print('MNIST encoder x: ',x.size())
		return x

	# 计算x的维度之积，拉成一条直线
	def num_flat_features(self,x):
		size = x.size()[1:] #除了batch之外的所有维度
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class Decoder_MNIST(nn.Module):
	def __init__(self,args):
		super(Decoder_MNIST,self).__init__()
		self.z_dim = args.z_dim
		self.fc1 = nn.Linear(self.z_dim,84)
		self.fc2 = nn.Linear(84,120)
		self.fc3 = nn.Linear(120,256)
		self.fc4 = nn.Linear(256,64*5*5)
		self.fc5 = nn.Linear(64*5*5,28*28)
	def forward(self,z):
		#print('MNIST decoder z: ',z.size())
		z = F.relu(self.fc1(z))
		#print('MNIST decoder z: ',z.size())
		z = F.relu(self.fc2(z))
		#print('MNIST decoder z: ',z.size())
		z = F.relu(self.fc3(z))
		#print('MNIST decoder z: ',z.size())
		z = F.relu(self.fc4(z))
		#print('MNIST decoder z: ',z.size())
		z = self.fc5(z)
		#print('MNIST decoder z: ',z.size())
		z = F.sigmoid(z)
		z = z.view(-1,1,28,28)
		# #print('z.size: ',z.size())
		return z

# 3Dchairs encoder and decoder
class Encoder_3Dchairs(nn.Module):
	def __init__(self,args):
		super(Encoder_3Dchairs,self).__init__()
		self.z_dim = args.z_dim
		self.conv0 = nn.Conv2d(1,16,4,2,1)
		self.conv1 = nn.Conv2d(16,32,4,2,1)
		self.conv2 = nn.Conv2d(32,32,4,2,1)
		self.conv3 = nn.Conv2d(32,64,4,2,1)
		self.conv4 = nn.Conv2d(64,64,4,2,1)
		self.conv5 = nn.Conv2d(64,256,4,1)

		self.fc = nn.Linear(256,self.z_dim*2)

		self.pool = nn.MaxPool2d(2,2)

	def forward(self,x):
		#print('3Dchairs encoder x: ',x.size())
		x = F.relu(self.conv0(x))
		#print('3Dchairs encoder x: ',x.size())
		x = F.relu(self.conv1(x))
		#print('3Dchairs encoder x: ',x.size())
		x = F.relu(self.conv2(x))
		#print('3Dchairs encoder x: ',x.size())
		x = F.relu(self.conv3(x))
		#print('3Dchairs encoder x: ',x.size())
		x = F.relu(self.conv4(x))
		#print('3Dchairs encoder x: ',x.size())
		x = F.relu(self.conv5(x))
		#print('3Dchairs encoder x: ',x.size())
		x = x.view(-1,self.num_flat_features(x))
		#print('3Dchairs encoder x: ',x.size())
		x = self.fc(x)
		#print('3Dchairs encoder x: ',x.size())
		return x

	# 计算x的维度之积，拉成一条直线
	def num_flat_features(self,x):
		size = x.size()[1:] #除了batch之外的所有维度
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class Decoder_3Dchairs(nn.Module):
	def __init__(self,args):
		super(Decoder_3Dchairs,self).__init__()
		self.z_dim = args.z_dim

		self.fc = nn.Linear(self.z_dim,256)
		self.deconv1 = nn.ConvTranspose2d(256, 64, 4)
		self.deconv2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
		self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
		self.deconv4 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
		self.deconv5 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
		self.deconv6 = nn.ConvTranspose2d(16, 1, 4, 2, 1)


	def forward(self,z):
		#print('3Dchairs decoder z: ',z.size())
		z = F.relu(self.fc(z))
		#print('3Dchairs decoder z: ',z.size())
		z = z.view(-1, 256, 1, 1)
		#print('3Dchairs decoder z: ',z.size())
		z = F.relu(self.deconv1(z))
		#print('3Dchairs decoder z: ',z.size())
		z = F.relu(self.deconv2(z))
		#print('3Dchairs decoder z: ',z.size())
		z = F.relu(self.deconv3(z))
		#print('3Dchairs decoder z: ',z.size())
		z = F.relu(self.deconv4(z))
		#print('3Dchairs decoder z: ',z.size())
		z = F.relu(self.deconv5(z))
		#print('3Dchairs decoder z: ',z.size())
		z = F.sigmoid(self.deconv6(z))
		#print('3Dchairs decoder z: ',z.size())
		return z

# dsprites encoder and decoder
class Encoder_dsprites(nn.Module):
	def __init__(self,args):
		super(Encoder_dsprites,self).__init__()
		self.z_dim = args.z_dim
		# self.conv0 = nn.Conv2d(1,16,4,2,1)
		self.conv1 = nn.Conv2d(1,32,4,2,1)
		self.conv2 = nn.Conv2d(32,32,4,2,1)
		self.conv3 = nn.Conv2d(32,64,4,2,1)
		self.conv4 = nn.Conv2d(64,64,4,2,1)
		self.conv5 = nn.Conv2d(64,256,4,1)

		self.fc = nn.Linear(256,self.z_dim*2)

		self.pool = nn.MaxPool2d(2,2)

	def forward(self,x):
		#print('dsprites encoder x: ',x.size())
		x = F.relu(self.conv1(x))
		#print('dsprites encoder x: ',x.size())
		x = F.relu(self.conv2(x))
		#print('dsprites encoder x: ',x.size())
		x = F.relu(self.conv3(x))
		#print('dsprites encoder x: ',x.size())
		x = F.relu(self.conv4(x))
		#print('dsprites encoder x: ',x.size())
		x = F.relu(self.conv5(x))
		#print('dsprites encoder x: ',x.size())
		x = x.view(-1,self.num_flat_features(x))
		#print('dsprites encoder x: ',x.size())
		x = self.fc(x)
		#print('dsprites encoder x: ',x.size())
		return x

	# 计算x的维度之积，拉成一条直线
	def num_flat_features(self,x):
		size = x.size()[1:] #除了batch之外的所有维度
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class Decoder_dsprites(nn.Module):
	def __init__(self,args):
		super(Decoder_dsprites,self).__init__()
		self.z_dim = args.z_dim

		self.fc = nn.Linear(self.z_dim,256)
		self.deconv1 = nn.ConvTranspose2d(256, 64, 4)
		self.deconv2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
		self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
		self.deconv4 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
		self.deconv5 = nn.ConvTranspose2d(32, 1, 4, 2, 1)


	def forward(self,z):
		#print('dsprites decoder z: ',z.size())
		z = F.relu(self.fc(z))
		#print('dsprites decoder z: ',z.size())
		z = z.view(-1, 256, 1, 1)
		#print('dsprites decoder z: ',z.size())
		z = F.relu(self.deconv1(z))
		#print('dsprites decoder z: ',z.size())
		z = F.relu(self.deconv2(z))
		#print('dsprites decoder z: ',z.size())
		z = F.relu(self.deconv3(z))
		#print('dsprites decoder z: ',z.size())
		z = F.relu(self.deconv4(z))
		#print('dsprites decoder z: ',z.size())
		z = F.sigmoid(self.deconv5(z))
		#print('dsprites decoder z: ',z.size())
		return z