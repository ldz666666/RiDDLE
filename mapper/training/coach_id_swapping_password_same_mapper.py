import os
import torch
import torchvision
from torch import nn, autograd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pprint
import json

from PIL import Image
import numpy as np

#sys path dependency
import sys
sys.path.append("/hd3/lidongze/animation/RiDDLE")
sys.path.append("/hd3/lidongze/animation/RiDDLE/mapper")
from torch.utils.tensorboard import SummaryWriter
from criteria import id_loss #loss1 id loss
from criteria.parse_related_loss import bg_loss
from criteria.landmark_loss import landmark_loss

from torch.utils.data import TensorDataset

from mapper.training.ranger import Ranger
from mapper.training import train_utils

from models.encoders.model_irse import Backbone
from models.stylegan2.model import Generator
from models.encoders.psp_encoders import GradualStyleEncoder,Encoder4Editing,BackboneEncoderUsingLastLayerIntoW
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator

from mapper import latent_id_mappers,transformer
from models.psp import get_keys,pSp

import logging 
#add by ldz
#from mapper.hairclip_mapper import HairCLIPMapper
from criteria import latent_loss
#just for testing
from mapper.options.train_options_pwd import TrainOptions

from train_utils import batch_tensor_to_np_row, RedirectLogger
from criteria.lpips.lpips import LPIPS

#ddp
import torch.distributed as dist
import math



class CoachID:
	def __init__(self, opts):
		self.opts = opts
		self.global_step = 0
		
		print('local rank',self.opts.local_rank)
		if not self.opts.ddp:
			self.device = "cuda"
			self.opts.device = self.device
		
		#distribution
		if self.opts.ddp:
			self.device=torch.device("cuda", self.opts.local_rank)
			self.opts.device = self.device
			dist.init_process_group(backend='nccl')
			#torch.cuda.set_device(self.opts.local_rank)

			print('dist related')
			print('dist world size',dist.get_world_size())
			print('dist rank',dist.get_rank())

		self.latents_num=(int(math.log(self.opts.stylegan_size,2))-1)*2
		print('latents_num',self.latents_num)
		# Initialize logger
		self.output_logger=self.configure_logger(os.path.join(self.opts.exp_dir,'record.log'))

		# Initialize networks 
		self.decoder=Generator(self.opts.stylegan_size,512,8).to(self.device)
		self.decoder.load_state_dict(torch.load(opts.stylegan_weights,map_location='cpu')['g_ema'],strict=False)
		self.facepool=nn.AdaptiveAvgPool2d((256,256)).to(self.device)
		
		self.arcface =  Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(self.device)
		self.arcface.load_state_dict(torch.load(opts.ir_se50_weights,map_location='cpu'))

		if self.opts.latent_augment:
			self.augmenter=train_utils.LatentAugmenter()
			print('augmenter loaded')

		#assert self.opts.mapper_type in ['simple','mask','transformer','selfattn']
		if self.opts.mapper_type=='simple':
			#self.mapper= latent_id_mappers.SimpleMapper().to(self.device)
			self.mapper= latent_id_mappers.SimpleMapper_morefc(1024,self.opts.morefc_num,self.opts.use_landmark_prior).to(self.device)
		elif self.opts.mapper_type=='transformer':
			#self.mapper = latent_id_mappers.TransformerMapper(normalize_type=self.opts.transformer_normalize_type).to(self.device)
			self.mapper = latent_id_mappers.TransformerMapperSplit(split_list=self.opts.transformer_split_list,normalize_type=self.opts.transformer_normalize_type,\
				add_linear=self.opts.transformer_add_linear,add_pos_embedding=self.opts.transformer_add_pos_embedding).to(self.device)


		print('print out mapper')
		print(self.mapper)
		print('print out mapper parameters')
		for name, paramer in self.mapper.named_parameters():
			print(name)
	
		#ddp module
		if self.opts.ddp:
			self.mapper = nn.parallel.DistributedDataParallel(self.mapper, device_ids=[self.opts.local_rank], output_device=self.opts.local_rank,broadcast_buffers=False, find_unused_parameters=True)
			
		#set arcface and stylegan decoder to eval
		self.arcface.eval() 
		self.decoder.eval()

		# Initialize losses
		
		self.lpips_loss = LPIPS(net_type=self.opts.lpips_type).to(self.device).eval()
		self.id_loss=id_loss.IDLossExtractor(self.opts,self.arcface).to(self.device).eval()
		self.latent_l2_loss = nn.MSELoss().to(self.device).eval()
		self.latent_cos_loss = nn.CosineEmbeddingLoss().to(self.device).eval()

		if self.opts.landmark_lambda > 0:
			self.landmark_loss = landmark_loss.LandmarkLoss(self.opts).to(self.device).eval()
		if self.opts.latent_align_lambda > 0:
			self.latent_align_loss = latent_loss.LatentAlignLoss(self.opts).to(self.device).eval()
		# if self.opts.background_lambda > 0: 
		# 	self.background_loss = bg_loss.BackgroundLoss(self.opts).to(self.device).eval()
		if self.opts.parse_lambda > 0: 
			self.parse_loss = bg_loss.ParseLoss(self.opts).to(self.device).eval()
		if self.opts.latent_avg_lambda > 0: #use latent avg
			self.latent_avg = self.decoder.mean_latent(int(1e5))[0].detach()
			self.latent_avg = self.latent_avg.unsqueeze(0).repeat(1,self.latents_num,1)
			print('latent avg shape',self.latent_avg.shape)
			self.latent_avg_loss = latent_loss.WNormLoss(start_from_latent_avg=True)

			
		# Initialize optimizer
		self.optimizer = self.configure_optimizers()
		'''
		random sampling latent codes
		'''

		# # Initialize dataset
		if self.opts.use_dataset:
			self.train_dataset, self.test_dataset = self.configure_datasets()
			
			#sampler
			train_sampler,test_sampler=None,None
			if self.opts.ddp:
				train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
				test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset)

			
			self.train_dataloader = DataLoader(self.train_dataset,
											batch_size=self.opts.batch_size,
											shuffle=(train_sampler is None),
											num_workers=int(self.opts.workers),
											sampler=train_sampler,
											drop_last=True)
			self.test_dataloader = DataLoader(self.test_dataset,
											batch_size=self.opts.test_batch_size,
											shuffle=False,
											sampler=test_sampler,
											num_workers=int(self.opts.test_workers),
											drop_last=True)
			self.iter_train=iter(self.train_dataloader)
			self.iter_test=iter(self.test_dataloader)
		# if self.opts.keep_optimizer:
		# 	self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

		# Initialize discriminator
		if self.opts.w_discriminator_lambda > 0:
			self.discriminator = LatentCodesDiscriminator(512, 4).to(self.device)
			self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
															lr=opts.w_discriminator_lr)
			self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
			self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)

			# if os.path.exists(self.opts.discriminator_weight):
			# 	self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
			# 	self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])
		
		# Initialize tensorboard
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.log_dir = log_dir
		self.logger = SummaryWriter(log_dir=log_dir)

		self.temp_img_dir = os.path.join(opts.exp_dir,'temp_images')
		os.makedirs(self.temp_img_dir,exist_ok=True)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

		if self.opts.resume_checkpoint_path is not None and self.opts.resume:
			prev_train_checkpoint = torch.load(self.opts.resume_checkpoint_path,map_location='cpu')
			#self.global_step=int(self.opts.resume_checkpoint_path.split('.')[-2].split('_')[-1])
			self.load_from_train_checkpoint(prev_train_checkpoint)
			prev_train_checkpoint = None

	def train(self):
		print('into training loop')
		#torch.set_printoptions(threshold=np.inf)
		self.mapper.train()
		while self.global_step < self.opts.max_steps:

			loss_dict_disc={}
			if self.is_training_discriminator():
				loss_dict_disc = self.train_discriminator()
			
			self.optimizer.zero_grad()
			
			if self.opts.use_dataset:
				w_ori=self.get_next_batch()[0] #latent dataset
				w_ori=w_ori.to(self.device)
			else:
				w_ori=self.sample_w()

			if self.opts.latent_augment:
				#print('augmenting')
				w_ori=self.augmenter.augment(w_ori)
			

			b=w_ori.shape[0]
			
			w_pwd1,w_pwd2=self.sample_w(),self.sample_w()
				
			w_cat1_enc=torch.cat([w_ori,w_pwd1],dim=-1)
			w_cat2_enc=torch.cat([w_ori,w_pwd2],dim=-1)

			
			with torch.no_grad():
				x_ori, _ = self.decoder([w_ori], input_is_latent=True, randomize_noise=False, truncation=1)
				if self.opts.use_landmark_prior and self.opts.landmark_lambda > 0:
					ldm_ori=self.landmark_loss.get_raw_output(x_ori)
				
			if self.opts.use_landmark_prior and self.opts.landmark_lambda > 0:
				w_enc1 = self.mapper(w_cat1_enc,ldm_ori)
				w_enc2 = self.mapper(w_cat2_enc,ldm_ori)
			else:
				w_enc1 = self.mapper(w_cat1_enc)
				w_enc2 = self.mapper(w_cat2_enc)

			if self.opts.latent_avg_lambda > 0:
				w_enc1=w_enc1+self.latent_avg
				w_enc2=w_enc2+self.latent_avg

			#enc latent code
			x_enc1, _ = self.decoder([w_enc1], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
			x_enc2, _ = self.decoder([w_enc2], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
			
			#recover process
			w_rand1,w_rand2=self.sample_w(),self.sample_w()


			if self.opts.use_landmark_prior and self.opts.landmark_lambda > 0:
				with torch.no_grad():
					ldm_enc1=self.landmark_loss.get_raw_output(x_enc1)


			
			
			w_cat1_dec=torch.cat([w_enc1,w_pwd1],dim=-1)
			w_cat1_dec_rand=torch.cat([w_enc1,w_rand1],dim=-1)
			w_cat2_dec_rand=torch.cat([w_enc1,w_rand2],dim=-1)

			
			if self.opts.use_landmark_prior and self.opts.landmark_lambda > 0:
				w_dec = self.mapper(w_cat1_dec,ldm_enc1)
				w_dec_rand1 = self.mapper(w_cat1_dec_rand,ldm_enc1)
				w_dec_rand2 = self.mapper(w_cat2_dec_rand,ldm_enc1)
			else:
				w_dec = self.mapper(w_cat1_dec)
				w_dec_rand1 = self.mapper(w_cat1_dec_rand)
				w_dec_rand2 = self.mapper(w_cat2_dec_rand)

			if self.opts.latent_avg_lambda > 0:
				w_dec=w_dec+self.latent_avg
				w_dec_rand1=w_dec_rand1+self.latent_avg
				w_dec_rand2=w_dec_rand2+self.latent_avg
			
			x_dec,_ = self.decoder([w_dec], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
			x_dec_rand1,_ = self.decoder([w_dec_rand1], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
			x_dec_rand2,_ = self.decoder([w_dec_rand2], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
			

			x_ori = self.facepool(x_ori) 
			x_enc1 = self.facepool(x_enc1) 
			x_enc2=self.facepool(x_enc2) 
			x_dec=self.facepool(x_dec) 
			x_dec_rand1=self.facepool(x_dec_rand1)
			x_dec_rand2=self.facepool(x_dec_rand2)

			if self.global_step%100==0:
				print(f"w_ori max {w_ori.max()} w_ori min {w_ori.min()} w_ori mean {w_ori.mean()}")
				print(f"w_enc1 max {w_enc1.max()} w_enc1 min {w_enc1.min()} w_enc1 mean {w_enc1.mean()}")
				print(f"w_dec max {w_dec.max()} w_dec min {w_dec.min()} w_dec mean {w_dec.mean()}")

			
			loss, loss_dict = self.calc_loss(w_ori, w_pwd1,w_pwd2, w_enc1,w_enc2,w_dec,w_dec_rand1,w_dec_rand2,\
				x_ori,x_enc1,x_enc2,x_dec,x_dec_rand1,x_dec_rand2)
			
			if self.is_training_discriminator():
				loss_dict = {**loss_dict, **loss_dict_disc}
			
			loss.backward()
			self.optimizer.step()

			if self.opts.parse_lambda > 0:
				with torch.no_grad():
					mask_ori=self.parse_loss.gen_mask(x_ori).repeat(1,3,1,1)
					mask_enc1=self.parse_loss.gen_mask(x_enc1).repeat(1,3,1,1)
					mask_dec=self.parse_loss.gen_mask(x_dec).repeat(1,3,1,1)
					mask_dec_rand1=self.parse_loss.gen_mask(x_dec_rand1).repeat(1,3,1,1)


			# Logging related
			if self.global_step % self.opts.image_interval == 0 or (
					self.global_step < 1000 and self.global_step % 1000 == 0):
				if (not self.opts.ddp) or (self.opts.ddp and dist.get_rank()==0): 
					self.parse_and_log_images(x_ori,x_enc1, x_enc2,x_dec,x_dec_rand1,x_dec_rand2) #may need to redefine
					if self.opts.parse_lambda > 0:
						self.parse_and_log_images(x_ori,mask_ori,x_enc1,mask_enc1,x_dec,mask_dec,x_dec_rand1,mask_dec_rand1,name='mask')
					#self.log_images_for_test() #may need to redefine
			
			if self.global_step % self.opts.board_interval == 0:
				self.print_metrics(loss_dict, prefix='train')
				self.log_metrics(loss_dict, prefix='train')

			# Validation related
			val_loss_dict = None

			if self.global_step % self.opts.save_interval == 0 or self.global_step+1 == self.opts.max_steps:
				self.checkpoint_me(loss_dict, is_best=False)

			if self.global_step == self.opts.max_steps:
				print('OMG, finished training!')
				break

			self.global_step += 1
			
	def load_from_train_checkpoint(self, ckpt):
		print('Loading previous training data...')
		self.global_step = ckpt['global_step'] + 1
		print(f'resume, global step {self.global_step}')

		self.mapper.load_state_dict(ckpt['mapper_state_dict'])

		if self.opts.keep_optimizer:
			self.optimizer.load_state_dict(ckpt['optimizer'])
		if self.opts.w_discriminator_lambda > 0:
			self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
			self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])

		print(f'Resuming training from step {self.global_step}')

	def checkpoint_me(self, loss_dict,is_best=False):
		save_name = 'iteration_{}.pt'.format(self.global_step)
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_optimizers(self):
		#set mappers parameter 
		params=[{'params': self.mapper.parameters()}]
		# if self.ddp:
		# 	self.opts.
		#params = list(self.mapper.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):

        #load train latents and test latents
		if self.opts.latents_train_path:
			train_latents = torch.load(self.opts.latents_train_path)
		else: 
			train_latents_z = torch.randn(self.opts.train_dataset_size, 512).cuda()
			train_latents = []
			for b in range(self.opts.train_dataset_size // self.opts.batch_size):
				with torch.no_grad():
					_, train_latents_b = self.decoder([train_latents_z[b: b + self.opts.batch_size]],
														  truncation=0.7, truncation_latent=None, return_latents=True)
					train_latents.append(train_latents_b)
			train_latents = torch.cat(train_latents)

		if self.opts.latents_test_path:
			test_latents = torch.load(self.opts.latents_test_path)
		else:
			test_latents_z = torch.randn(self.opts.train_dataset_size, 512).cuda()
			test_latents = []
			for b in range(self.opts.test_dataset_size // self.opts.test_batch_size):
				with torch.no_grad():
					_, test_latents_b = self.decoder([test_latents_z[b: b + self.opts.test_batch_size]],
													  truncation=0.7, truncation_latent=None, return_latents=True)
					test_latents.append(test_latents_b)
			test_latents = torch.cat(test_latents)

		if self.opts.id_emb_train_path:
			train_id_emb=torch.load(self.opts.id_emb_train_path)
		else:
			raise Exception('train id embs not set')

		if self.opts.id_emb_test_path:
			test_id_emb=torch.load(self.opts.id_emb_test_path)
		else:
			raise Exception('test id embs not set')

		# train_dataset = LatentsPairDataset(latents=train_latents.cpu(),
		#                                       opts=self.opts,
		#                                       )
		# test_dataset = LatentsPairDataset(latents=test_latents.cpu(),
		#                                       opts=self.opts,
		#                                       )
		train_dataset = TensorDataset(train_latents)
		test_dataset = TensorDataset(test_latents)

		
		print("Number of training samples: {}".format(len(train_dataset)), flush=False)
		print("Number of test samples: {}".format(len(test_dataset)), flush=False)
		return train_dataset, test_dataset


	def configure_logger(self,logging_name):
		if not self.opts.ddp:
			sys.stdout = RedirectLogger(logging_name)
		else:
			if dist.get_rank()==0:
				sys.stdout = RedirectLogger(logging_name)
			else:
				sys.stdout = open(os.devnull, 'w')

	def calc_loss(self,w_ori, w_pwd1,w_pwd2, w_enc1,w_enc2,w_dec,w_dec_rand1,w_dec_rand2,\
				x_ori,x_enc1,x_enc2,x_dec,x_dec_rand1,x_dec_rand2):

		
		b=x_ori.shape[0]

		loss_dict = {}
		loss = 0.0

		if self.is_training_discriminator():  # Adversarial loss for g
			loss_disc = 0
			
			w_cat_for_disc=torch.cat([w_enc1,w_enc2,w_dec,w_dec_rand1,w_dec_rand2],dim=0)
			for i in range(w_cat_for_disc.shape[1]):
				w = w_cat_for_disc[:, i, :]
				fake_pred = self.discriminator(w)
				loss_disc += F.softplus(-fake_pred).mean()
			loss_disc = loss_disc/w_cat_for_disc.shape[1]
			loss_dict['loss_mapper_disc'] = float(loss_disc)
			loss += self.opts.w_discriminator_lambda * loss_disc

		#lpips loss: all the image should be similar with w_ori
		if self.opts.lpips_lambda > 0:
			loss_lpips = 0 
			loss_lpips += self.lpips_loss(x_ori, x_enc1)
			loss_lpips += self.lpips_loss(x_ori, x_enc2)
			loss_lpips += self.lpips_loss(x_ori, x_dec)
			loss_lpips += self.lpips_loss(x_ori, x_dec_rand1)
			loss_lpips += self.lpips_loss(x_ori, x_dec_rand2)
			loss_dict['loss_lpips'] = float(loss_lpips)* self.opts.lpips_lambda
			loss += loss_lpips * self.opts.lpips_lambda
		
		if self.opts.landmark_lambda > 0:
			loss_landmark = 0
			loss_landmark +=self.landmark_loss(x_ori, x_enc1)
			loss_landmark +=self.landmark_loss(x_ori, x_enc2)
			loss_landmark +=self.landmark_loss(x_ori, x_dec)
			loss_landmark +=self.landmark_loss(x_ori, x_dec_rand1)
			loss_landmark +=self.landmark_loss(x_ori, x_dec_rand2)
			loss_landmark = loss_landmark / 5
			loss_dict['loss_landmark'] = float(loss_landmark)* self.opts.landmark_lambda
			loss += loss_landmark * self.opts.landmark_lambda

		#de id loss enc
		if self.opts.rev_id_enc_lambda>0:

			loss_rev_id_enc=0
			_, loss_rev_id_enc1, _= self.id_loss(x_ori,x_enc1)
			_, loss_rev_id_enc2, _= self.id_loss(x_ori,x_enc2)
			_, loss_rev_id_enc12, _= self.id_loss(x_enc1,x_enc2)
			loss_rev_id_enc = loss_rev_id_enc1 + loss_rev_id_enc2 + self.opts.rev_id_intra_lambda*loss_rev_id_enc12
			loss_rev_id_enc = loss_rev_id_enc/(3-1+self.opts.rev_id_intra_lambda)
			loss_dict['loss_rev_id_enc'] = float(loss_rev_id_enc) * self.opts.rev_id_enc_lambda
			loss += loss_rev_id_enc * self.opts.rev_id_enc_lambda
		#rev latent loss for encoding
		if self.opts.rev_latent_enc_lambda>0:

			loss_rev_latent_enc=0
			loss_rev_latent_enc += torch.cosine_similarity(w_ori,w_enc1,dim=-1).mean()
			loss_rev_latent_enc += torch.cosine_similarity(w_ori,w_enc2,dim=-1).mean()
			loss_rev_latent_enc += torch.cosine_similarity(w_enc1,w_enc2,dim=-1).mean()
			loss_rev_latent_enc = loss_rev_latent_enc/3
			loss_dict['loss_rev_latent_enc'] = float(loss_rev_latent_enc) * self.opts.rev_latent_enc_lambda
			loss += loss_rev_latent_enc * self.opts.rev_latent_enc_lambda
		#latent loss
		if self.opts.latent_l2_enc_lambda > 0:
			loss_latent_l2_enc = 0
			loss_latent_l2_enc += (w_enc1**2).mean()
			loss_latent_l2_enc += (w_enc2**2).mean()
			loss_latent_l2_enc = loss_latent_l2_enc/2
			loss_dict['loss_latent_l2_enc'] = float(loss_latent_l2_enc)* self.opts.latent_l2_enc_lambda
			loss += loss_latent_l2_enc * self.opts.latent_l2_enc_lambda

		#de id loss dec
		if self.opts.rev_id_dec_lambda>0:

			loss_rev_id_dec=0
			_, loss_rev_id_dec1, _= self.id_loss(x_dec,x_dec_rand1)
			_, loss_rev_id_dec2, _= self.id_loss(x_dec,x_dec_rand2)
			_, loss_rev_id_dec12, _= self.id_loss(x_dec_rand1,x_dec_rand2)
			_, loss_rev_id_dec1ori, _ = self.id_loss(x_ori,x_dec_rand1)
			_, loss_rev_id_dec2ori, _ = self.id_loss(x_ori,x_dec_rand2)

			# if self.opts.rev_dec_enc:
			# 	_, loss_rev_id_de1, _= self.id_loss(x_enc1,x_dec_rand1)
			# 	_, loss_rev_id_de2, _= self.id_loss(x_enc1,x_dec_rand2)
			# 	loss_rev_id_dec = loss_rev_id_dec1 + loss_rev_id_dec2 + loss_rev_id_dec12 +  loss_rev_id_dec1ori + loss_rev_id_dec2ori \
			# 		+ loss_rev_id_de1 + loss_rev_id_de2
			# 	loss_rev_id_dec = loss_rev_id_dec/7
			
			loss_rev_id_dec = loss_rev_id_dec1 + loss_rev_id_dec2 + loss_rev_id_dec12*self.opts.rev_id_intra_lambda +  loss_rev_id_dec1ori + loss_rev_id_dec2ori
			loss_rev_id_dec = loss_rev_id_dec/(5-1+self.opts.rev_id_intra_lambda)
			loss_dict['loss_rev_id_dec'] = float(loss_rev_id_dec) * self.opts.rev_id_dec_lambda
			loss += loss_rev_id_dec * self.opts.rev_id_dec_lambda

		if self.opts.id_div_lambda > 0:

			loss_id_div=0
			id_enc1=self.id_loss.extract_feats(x_enc1)
			id_enc2=self.id_loss.extract_feats(x_enc2)
			id_dec_rand1=self.id_loss.extract_feats(x_dec_rand1)
			id_dec_rand2=self.id_loss.extract_feats(x_dec_rand2)
			id_bank=torch.cat([id_enc1,id_enc2,id_dec_rand1,id_dec_rand2],dim=0) #[4*b,512]
			id_mat=torch.mm(id_bank,id_bank.t())
			coe_mat=torch.ones_like(id_mat)-torch.eye(id_mat.shape[0]).to(id_mat.device)
			loss_id_div=coe_mat*id_mat
			loss_id_div=loss_id_div.mean()
			loss_dict['loss_id_div'] = float(loss_id_div)*self.opts.id_div_lambda 
			loss+=float(loss_id_div)*self.opts.id_div_lambda 
	

		#rev latent loss dec
		if self.opts.rev_latent_dec_lambda>0:

			loss_rev_latent_dec=0
			loss_rev_latent_dec += torch.cosine_similarity(w_dec,w_dec_rand1,dim=-1).mean()
			loss_rev_latent_dec += torch.cosine_similarity(w_dec,w_dec_rand2,dim=-1).mean()
			loss_rev_latent_dec += torch.cosine_similarity(w_dec_rand1,w_dec_rand2,dim=-1).mean()
			loss_rev_latent_dec += torch.cosine_similarity(w_ori,w_dec_rand1,dim=-1).mean()
			loss_rev_latent_dec += torch.cosine_similarity(w_ori,w_dec_rand2,dim=-1).mean()
			loss_rev_latent_dec = loss_rev_latent_dec/5
			loss_dict['loss_rev_latent_dec'] = float(loss_rev_latent_dec) * self.opts.rev_latent_dec_lambda
			loss += loss_rev_latent_dec * self.opts.rev_latent_dec_lambda
		#reconstruction loss
		if self.opts.recon_latent_lambda > 0:
			loss_recon_latent = F.mse_loss(w_dec,w_ori)
			loss_dict['loss_recon_latent'] = float(loss_recon_latent)* self.opts.recon_latent_lambda
			loss += loss_recon_latent * self.opts.recon_latent_lambda

		if self.opts.recon_id_lambda > 0:
			loss_recon_id, _, _= self.id_loss(x_dec,x_ori)
			loss_dict['loss_recon_id'] = float(loss_recon_id)* self.opts.recon_id_lambda
			loss += loss_recon_id * self.opts.recon_id_lambda
			
		if self.opts.recon_pix_lambda > 0:
			loss_recon_pix=0
			loss_recon_pix += F.l1_loss(x_ori,x_enc1)
			loss_recon_pix += F.l1_loss(x_ori,x_enc2)
			loss_recon_pix += F.l1_loss(x_ori,x_dec)
			loss_recon_pix += F.l1_loss(x_ori,x_dec_rand1)
			loss_recon_pix += F.l1_loss(x_ori,x_dec_rand2)
			loss_dict['loss_recon_pix'] = float(loss_recon_pix)*self.opts.recon_pix_lambda
			loss += loss_recon_pix*self.opts.recon_pix_lambda
		#latent l2 loss
		if self.opts.latent_l2_dec_lambda > 0:
			loss_latent_l2_dec = 0
			loss_latent_l2_dec += (w_dec**2).mean()
			loss_latent_l2_dec += (w_dec_rand1**2).mean()
			loss_latent_l2_dec += (w_dec_rand2**2).mean()
			loss_latent_l2_dec = loss_latent_l2_dec/3
			loss_dict['loss_latent_l2_dec'] = float(loss_latent_l2_dec)* self.opts.latent_l2_dec_lambda
			loss += loss_latent_l2_dec * self.opts.latent_l2_dec_lambda

		if self.opts.latent_align_lambda > 0:
			loss_latent_align = 0
			loss_latent_align += self.latent_align_loss(w_ori,w_enc1)
			loss_latent_align += self.latent_align_loss(w_ori,w_enc2)
			loss_latent_align += self.latent_align_loss(w_ori,w_dec)
			loss_latent_align += self.latent_align_loss(w_ori,w_dec_rand1)
			loss_latent_align += self.latent_align_loss(w_ori,w_dec_rand2)
			loss_dict['loss_latent_align'] = float(loss_latent_align)* self.opts.latent_align_lambda
			loss += loss_latent_align * self.opts.latent_align_lambda
		
		if self.opts.parse_lambda > 0: 
			loss_parse = 0
			loss_parse += self.parse_loss(x_ori,x_enc1)
			loss_parse += self.parse_loss(x_ori,x_enc2)
			loss_parse += self.parse_loss(x_ori,x_dec)
			loss_parse += self.parse_loss(x_ori,x_dec_rand1)
			loss_parse += self.parse_loss(x_ori,x_dec_rand2)
			loss_dict['loss_parse'] = float(loss_parse)* self.opts.parse_lambda
			loss += loss_parse * self.opts.parse_lambda

		if self.opts.latent_avg_lambda > 0:
			loss_latent_avg = self.latent_avg_loss(w_enc1, self.latent_avg)
			loss_latent_avg = self.latent_avg_loss(w_enc2, self.latent_avg)
			loss_latent_avg = self.latent_avg_loss(w_dec, self.latent_avg)
			loss_latent_avg = self.latent_avg_loss(w_dec_rand1, self.latent_avg)
			loss_latent_avg = self.latent_avg_loss(w_dec_rand2, self.latent_avg)
			loss_dict['loss_latent_avg'] = float(loss_latent_avg) * self.opts.latent_avg_lambda
			loss += loss_latent_avg * self.opts.latent_avg_lambda
		

		loss_dict['loss'] = float(loss)

		return loss, loss_dict

	#log metrics: add all the losses into tensorboard
	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print('Metrics for {}, step {}\t'.format(prefix, self.global_step), flush=False,end="")
		for key, value in metrics_dict.items():
			print('\t{} = {:.6f}'.format(key, value), flush=False,end="")
		print()

	def get_next_batch(self,training=True):
		if training:
			try:
				batch = next(self.iter_train)
			except StopIteration as e: 
				self.iter_train = iter(self.train_dataloader)
				batch = next(self.iter_train)
		else:
			try:
				batch = next(self.iter_test)
			except StopIteration as e: 
				self.iter_test = iter(self.test_dataloader)
				batch = next(self.iter_test)

		return batch


	def sample_w(self):
		z = torch.randn(self.opts.batch_size, 512, device=self.device)
		w = self.decoder.style(z).unsqueeze(1).repeat(1,self.latents_num,1)
		return w

	def sample_w_raw(self,b,repeat=False):
		z = torch.randn(b, 512, device=self.device)
		w = self.decoder.style(z).unsqueeze(1)
		if repeat:
			w=w.repeat(1,self.latents_num,1)
		return w

	def sample_random_code(self):
		w_enc1,w_ori,w_rand = self.sample_w(),self.sample_w(),self.sample_w()
		return w_ori, w_enc1, w_rand

	def reference_editing_interfacegan(self,w1,w2,b): #edit w2 according to w1
		return w2-((w2*b).sum(dim=-1,keepdim=True))*b+((w1*b).sum(dim=-1,keepdim=True))*b

	
	def parse_and_log_images(self,*args,name='pil'):
		
		print('into parse and log images, global step',self.global_step)
		# print(len(args))
		# print(type(args))
		save_total_list=[]
		for img in args:
			save_total_list.append(batch_tensor_to_np_row(img.detach().cpu()))
		save_total_np=np.concatenate(save_total_list,axis=0)
		Image.fromarray(save_total_np).save(os.path.join(self.temp_img_dir,f"{self.global_step:0>5d}_{name}.png"))

	def __get_save_dict(self):
		save_dict = {
			'mapper_state_dict': self.mapper.state_dict(),
			'opts': vars(self.opts)
		}
		if self.opts.save_training_data:
			save_dict['global_step'] = self.global_step
			save_dict['optimizer'] = self.optimizer.state_dict()
			if self.opts.w_discriminator_lambda > 0:
				save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
				save_dict['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
		return save_dict

	#discriminator related
	def is_training_discriminator(self):
		return self.opts.w_discriminator_lambda > 0
	
	@staticmethod
	def discriminator_loss(real_pred, fake_pred, loss_dict):
		real_loss = F.softplus(-real_pred).mean()
		fake_loss = F.softplus(fake_pred).mean()

		loss_dict['d_real_loss'] = float(real_loss)
		loss_dict['d_fake_loss'] = float(fake_loss)

		return real_loss + fake_loss

	@staticmethod
	def discriminator_r1_loss(real_pred, real_w):
		grad_real, = autograd.grad(
			outputs=real_pred.sum(), inputs=real_w, create_graph=True
		)
		grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

		return grad_penalty

	@staticmethod
	def requires_grad(model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag

	def train_discriminator(self):
		loss_dict = {}
		self.requires_grad(self.discriminator, True)
		with torch.no_grad():
			real_w,fake_w=self.sample_real_and_fake_latents()
		loss=0
		for i in range(real_w.shape[1]):
			real_pred = self.discriminator(real_w[i])
			fake_pred = self.discriminator(fake_w[i])
			loss += self.discriminator_loss(real_pred, fake_pred, loss_dict)
		loss = loss / real_w.shape[1]
		loss_dict['discriminator_loss'] = float(loss)

		self.discriminator_optimizer.zero_grad()
		loss.backward()
		self.discriminator_optimizer.step()

		# Reset to previous state
		self.requires_grad(self.discriminator, False)

		return loss_dict

	def sample_real_and_fake_latents(self):
		w_ori,w_pwd1,w_pwd2=self.sample_w(),self.sample_w(),self.sample_w()
		w_wrong_pwd1,w_wrong_pwd2=self.sample_w(),self.sample_w()
		w_cat_enc1=torch.cat([w_ori,w_pwd1],dim=-1)
		w_cat_enc2=torch.cat([w_ori,w_pwd2],dim=-1)
		w_enc1=self.mapper(w_cat_enc1)
		w_enc2=self.mapper(w_cat_enc2)

		w_dec=self.mapper(torch.cat([w_enc1,w_pwd1],dim=-1))
		w_dec_wrong1=self.mapper(torch.cat([w_enc1,w_wrong_pwd1],dim=-1))
		w_dec_wrong2=self.mapper(torch.cat([w_enc1,w_wrong_pwd2],dim=-1))

		fake_w=torch.cat([w_enc1,w_enc2,w_dec,w_dec_wrong1,w_dec_wrong2],dim=0)
		real_w=self.sample_w_raw(fake_w.shape[0],repeat=True) #[b,18,512]

		return real_w, fake_w



if __name__=="__main__":
	opts = TrainOptions().parse() #get parse_args
	print('local rank',opts.local_rank)
	os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = CoachID(opts)
	coach.train()