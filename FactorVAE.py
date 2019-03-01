import argparse
from FactorVAE_dataset import return_data
from FactorVAE_model import Encoder_MNIST,Decoder_MNIST,Encoder_3Dchairs,Decoder_3Dchairs
from FactorVAE_show import showimg,travel_img_showimg, show_active_units
from FactorVAE_solver import Solver


import warnings
warnings.filterwarnings("ignore")

def main(args):
	net = Solver(args)
	if args.istrain:
		net.train()
		net.loadmdoel_travese()
	else:
		net.loadmdoel_travese()

	# test dsprites dataset 
	# print('datasetname: ',args.datasetname)
	# train_loader = return_data(args)
	# for i,img in enumerate(train_loader,0):
	# 	print('i: ',i,' img: ',img.size())

if __name__ == "__main__":
	# local dir
	FactorVAE_result_path_local = 'E:/pytorch_FactorVAE_result/'
	data_dir_local = {'MNIST':'E:/DATA/MNIST/data',
					  '3Dchairs':'E:/DATA/',
					  'dsprites':'E:/DATA/'}
	result_path_all_dataset_local = {'MNIST':FactorVAE_result_path_local+'MNIST/MNIST_result/',
									'3Dchairs':FactorVAE_result_path_local+'3Dchairs/3Dchairs_result/',
									'dsprites':FactorVAE_result_path_local+'dsprites/dsprites_result/'}

	# server dir
	FactorVAE_result_path_server = '../../result/pytorch_FactorVAE_result/'
	data_dir_server = {'MNIST':'../../dataset/MNIST/data',
					   '3Dchairs':'../../dataset/',
					   'dsprites':'../../dataset/'}
	result_path_all_dataset_server = {'MNIST':FactorVAE_result_path_server+'MNIST/MNIST_result/',
									  '3Dchairs':FactorVAE_result_path_server+'3Dchairs/3Dchairs_result/',
									  'dsprites':FactorVAE_result_path_server+'dsprites/dsprites_result/'}

	parser = argparse.ArgumentParser(description='liqiang Beta-VAE')
	# 在 local 还是 server 上
	parser.add_argument('--local', default=True, help='True->local , False->server')
	# directories
	parser.add_argument('--data_dir', default=data_dir_server, help='data directory')
	parser.add_argument('--result_path_all_dataset', default=result_path_all_dataset_server, help='train result directory')
	parser.add_argument('--datasetname', default='MNIST', type=str, help='dataset name')
	
	# load model 
	parser.add_argument('--load_model_path', default='', help='load model path')
	# load data
	parser.add_argument('--num_workers', default=3, help='dataloader numworkers')
	parser.add_argument('--shuffle', default=False, type=bool, help='data load shuffle')
	
	# train
	parser.add_argument('--tc_gamma_weight', default=6.4, type=float, help='beta weight')
	parser.add_argument('--istrain',default=True,type=bool,help='if train->True, else ->False')

	parser.add_argument('--z_dim', default=32, type=int, help='dimension of the representation z')
	parser.add_argument('--image_size', default=128, type=int, help='image size for 3dchairs')
	parser.add_argument('--batchszie', default=16, type=int, help='batch size')
	parser.add_argument('--learning_rate', default=1e-4, type=float, help='encoder decoder learning rate')
	parser.add_argument('--train_epoch', default=50, type=int, help='train epoch')
	parser.add_argument('--show_img_step', default=1000, type=int, help='after how many steps show iamges')
	parser.add_argument('--save_model_step', default=2000, type=int, help='after how many steps save model')
	# z travese 
	parser.add_argument('--z_travese_sample_imgth', default=88, type=int, help='z travese with which image')
	parser.add_argument('--z_travese_limit', default=2, type=int, help='z travese limit(dimension max and min)')
	parser.add_argument('--z_travese_interval', default=0.1, type=float, help='z travese interval(dimension interval)')
	parser.add_argument('--z_travese_number_per_line', default=10, type=int, help='z travese number displayed per line')
	parser.add_argument('--var_threshold', default=5e-2, type=float, help='z active units threshold')
	args = parser.parse_args()

	args.local = True
	args.datasetname = 'MNIST'
	args.istrain = False
	args.z_travese_sample_imgth = 66
	# print('args.z_travese_sample_imgth: ',args.z_travese_sample_imgth)

	if args.local: #在local
		args.data_dir = data_dir_local
		args.result_path_all_dataset = result_path_all_dataset_local
	
	# 定义saved model的地址
	args.load_model_path = 'E:/pytorch_FactorVAE_result/MNIST/MNIST_result_gamma6.4_zdim10_server/'

	# 根据不同的数据集进行定义不同的参数值
	if args.datasetname == 'MNIST':
		args.z_dim = 10
		args.image_size = 28
		args.train_epoch = 50
	
	elif args.datasetname == '3Dchairs':
		args.z_dim = 32
		args.image_size = 128
		args.train_epoch = 50
	
	elif args.datasetname == 'dsprites':
		args.z_dim = 10
		args.image_size = 64
		args.train_epoch = 50

	
	
	
	main(args)