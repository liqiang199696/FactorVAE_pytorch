"""dataset.py"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder,MNIST
from torchvision import transforms

import gc


class CustomImageFolder(ImageFolder):
	def __init__(self, root, transform=None):
		super(CustomImageFolder, self).__init__(root, transform)

	def __getitem__(self, index):
		path = self.imgs[index][0]
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)
		return img


class CustomTensorDataset(Dataset):
	def __init__(self, data_tensor):
		self.data_tensor = data_tensor

	def __getitem__(self, index):
		return self.data_tensor[index]

	def __len__(self):
		return self.data_tensor.size(0)

def return_data(args):
	
	datasetname=args.datasetname
	dset_dir=args.data_dir[args.datasetname]
	batch_size=args.batchszie * 2
	num_workers=args.num_workers
	image_size=args.image_size
	shuffle = args.shuffle
	# assert image_size == 64, 'currently only image size of 64 is supported'

	if datasetname.lower() == 'mnist':
		# root = os.path.join(dset_dir, 'MNIST')
		root = dset_dir
		transform = transforms.Compose([transforms.ToTensor(),])
		trainset=MNIST(root,train=True,transform=transform,download=False)
		train_loader=DataLoader(trainset,
								batch_size=batch_size,
								shuffle=shuffle,
								num_workers=num_workers)
		return train_loader

	elif datasetname.lower() == '3dchairs':
		root = os.path.join(dset_dir, '3Dchairs/rendered_chairs')
		transform = transforms.Compose([
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),])
		train_kwargs = {'root':root, 'transform':transform}
		train_data = CustomImageFolder(**train_kwargs)
		train_loader = DataLoader(train_data,
								  batch_size=batch_size,
								  shuffle=shuffle,
								  num_workers=num_workers,
								  pin_memory=True,
								  drop_last=True)
		return train_loader

	elif datasetname.lower() == 'dsprites':
		# 数据集地址
		root = os.path.join(dset_dir, 'dsprites/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
		# if not os.path.exists(root):
		#	 import subprocess
		#	 print('Now download dsprites-dataset')
		#	 subprocess.call(['./download_dsprites.sh'])
		#	 print('Finished')
		data = np.load(root, encoding='bytes')
		# 因为内存不够，就先那就拿来一部分数据
		data_tensor = torch.from_numpy(data['imgs'][0:200001]).unsqueeze(1).float()
		# 释放data的内存
		del data
		gc.collect()
		
		train_kwargs = {'data_tensor':data_tensor}
		train_data = CustomTensorDataset(**train_kwargs)
		train_loader = DataLoader(train_data,
								  batch_size=batch_size,
								  shuffle=shuffle,
								  num_workers=num_workers,
								  pin_memory=False ,
								  drop_last=True)
		return train_loader


	else:
		raise NotImplementedError


	


