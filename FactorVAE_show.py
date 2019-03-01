# 在服务器上要加
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import torch
from torch.autograd import Variable
import gc

def show_active_units(z_dim, dataset_loader, datasetname, encoder, var_threshold):
	image_size = 64
	N = len(dataset_loader.dataset)  # number of data samples
	K = z_dim				 # number of latent variables
	# nparams = 2 #vae.q_dist.nparams
	# vae.eval()

	# print('Computing q(z|x) distributions.')
	qz_params = torch.Tensor(N, K)

	n = 0
	for xs in dataset_loader:
		if datasetname == 'MNIST':
			image_size = 28
			xs = xs[0]
		batch_size = xs.size(0)
		xs = Variable(xs.view(batch_size, 1, image_size, image_size).cuda(), volatile=True)
		z = encoder(xs)
		# print('z size: ',z.size())
		qz_params[n:n + batch_size] = z[:,0:z_dim].data
		n += batch_size

	del dataset_loader
	gc.collect()

	var = torch.std(qz_params.contiguous().view(N, K), dim=0).pow(2)
	print('var: ',var)
	# print('var')
	# print(var)
	# 使用方差来判断是不是大于阈值来进行判断该维度中是不是有信息
	active_units = torch.arange(0, K)[var > var_threshold].long()
	print('Active units: ' + ','.join(map(str, active_units.tolist())))
	n_active = len(active_units)
	print('Number of active units: {}/{}'.format(n_active, z_dim))



def showimg(images,train_counter,images_classes,datasetname,generated_images_path,show_img_step):
	images=images.detach().cpu().numpy()
	# print('showing images: ',images.shape)
	
	if datasetname == 'MNIST':
		images=255*(0.5*images+0.5)
		images = images.astype(np.uint8)
	elif datasetname == '3Dchairs':
		# 彩色时用
		# images = images.transpose([0,2,3,1])
		pass

	grid_length=int(np.ceil(np.sqrt(images.shape[0])))
	# print('grid_length: ',grid_length)
	plt.figure(figsize=(2*grid_length,2*grid_length))
	gs = gridspec.GridSpec(grid_length,grid_length,wspace=0,hspace=0.1)
	# gs.update(wspace=0, hspace=0)
	# print('starting...')
	for i, img in enumerate(images):
		ax = plt.subplot(gs[i])
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(img,cmap = plt.cm.gray)
		plt.axis('off')
		plt.tight_layout()
	# print('showing...')
	plt.tight_layout()
	plt.savefig(generated_images_path+'%d'%(train_counter/show_img_step)+'_%s'%images_classes+'.png', bbox_inches='tight')
	


def travel_img_showimg(images,ith,limit_count,datasetname,travese_z_path,z_travese_number_per_line):
	images=images.detach().cpu().numpy()
	if datasetname == 'MNIST':
		images=255*(0.5*images+0.5)
		images = images.astype(np.uint8)
	elif datasetname == '3Dchairs':
		pass
	# print('travel_img_showimg: ',images.shape)
	grid_length=int(np.ceil(images.shape[0]))
	plt.figure(figsize=(3*z_travese_number_per_line,3*int(limit_count/z_travese_number_per_line)))
	width = int((images.shape[2]))
	gs = gridspec.GridSpec(int(limit_count/z_travese_number_per_line),z_travese_number_per_line,wspace=0,hspace=0.1)
	for i, img in enumerate(images):
		ax = plt.subplot(gs[i])
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(img.reshape([width,width]),cmap = plt.cm.gray)
		plt.axis('off')
		plt.tight_layout()
	plt.tight_layout()
	plt.savefig(travese_z_path+'sample_travese_dim_'+str(ith)+'.png', bbox_inches='tight')
