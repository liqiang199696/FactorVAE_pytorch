# 在服务器上要加
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec




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
