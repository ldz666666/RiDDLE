import os
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

#sys path dependency
import sys
sys.path.append("/hd3/lidongze/animation/RiDDLE")
sys.path.append("/hd3/lidongze/animation/RiDDLE/mapper")

from torch.utils.data import TensorDataset
from models.encoders.model_irse import Backbone
from models.stylegan2.model import Generator
from mapper import latent_id_mappers
from models.psp import get_keys,pSp
from models import seg_model_2

import logging 
from argparse import Namespace
from mapper.datasets.simple_dataset import SimpleDataset
#just for testing
from mapper.options.test_options_pwd import TestOptions
from criteria.id_loss import IDLossExtractor

from os.path import join

def reorder(ts,batch_size,range_num):

    index=torch.arange(ts.shape[0]).view(range_num,batch_size)
    index=index.T.reshape(batch_size,range_num).flatten()
    return ts[index]


#test engine
class CoachTest:
    def __init__(self,opts):
        #model related
        self.opts=opts
        self.device=self.opts.device
        self.e4e=self.load_e4e()
        self.decoder=Generator(self.opts.stylegan_size,512,8).to(self.device)
        self.decoder.load_state_dict(torch.load(self.opts.stylegan_weights,map_location='cpu')['g_ema'],strict=False)
        if self.opts.mapper_type=='simple':
            #self.mapper= latent_id_mappers.SimpleMapper().to(self.device)
            self.mapper= latent_id_mappers.SimpleMapper_morefc(1024).to(self.device)
        elif self.opts.mapper_type=='transformer':
            self.mapper = latent_id_mappers.TransformerMapperSplit(split_list=self.opts.transformer_split_list,normalize_type=self.opts.transformer_normalize_type,\
				add_linear=self.opts.transformer_add_linear,add_pos_embedding=self.opts.transformer_add_pos_embedding).to(self.device)
        self.mapper.load_state_dict(torch.load(opts.mapper_weight,map_location='cpu')['mapper_state_dict'])
        #load segmentation model
        self.segmentation_model = seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
        self.segmentation_model.load_state_dict(torch.load("/hd3/lidongze/animation/STIT/pretrained_models/79999_iter.pth"))
        #load id extractor
        self.arcface =  Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(self.device)
        self.arcface.load_state_dict(torch.load(opts.ir_se50_weights,map_location='cpu'))
        self.id_loss=IDLossExtractor(self.opts,self.arcface).to(self.device).eval()
        #tensor dataset related
        self.latent=torch.load(opts.latent_path,map_location='cpu')
        self.latent_dataset = TensorDataset(self.latent)
        self.latent_loader = torch.utils.data.DataLoader(
                self.latent_dataset,
                batch_size=self.opts.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )
        #image dataset related
        trans=transforms.Compose([
                    transforms.Resize((self.opts.image_size, self.opts.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.image_dataset = SimpleDataset(self.opts.image_path,transform=trans)
        self.image_loader = torch.utils.data.DataLoader(
                self.image_dataset,
                batch_size=self.opts.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )
        #working dir
        self.exp_dir=self.opts.exp_dir
        os.makedirs(self.exp_dir,exist_ok=True)
        #pwd list
        self.w_pwd_list=[self.sample_w(1) for i in range(self.opts.pwd_num)]
        self.w_wrong_pwd_list=[self.sample_w(1) for i in range(self.opts.pwd_num)]
        
    def load_e4e(self):
        print('loading e4e')
        print(self.opts.e4e_model_weights)
        ckpt = torch.load(self.opts.e4e_model_weights, map_location='cpu')
        opts = ckpt['opts'] #may need to see the opt
        opts['checkpoint_path'] = self.opts.e4e_model_weights
        opts['stylegan_weights'] = self.opts.stylegan_weights
        print(opts) #see the opts
        opts= Namespace(**opts)
        net = pSp(opts)
        print(f'loaded e4e from {self.opts.e4e_model_weights}')
        print(f'e4e stylegan is from { self.opts.stylegan_weights}')
        return net.to(self.device)

    def sample_w(self,b):
        z = torch.randn(b, 512, device=self.device)
        w = self.decoder.style(z).unsqueeze(1).repeat(1,self.opts.latents_num,1)
        return w

    def single_pwd_loop(self):
        output_frame_path=join(self.exp_dir,'frames')
        output_video_path=join(self.exp_dir,f'{self.exp_dir.split("/")[-1]}.mp4')
        os.makedirs(output_frame_path,exist_ok=True)
        w_pwd=self.w_pwd_list[0]
        print(w_pwd.shape)
        w_wrong_pwd=self.w_wrong_pwd_list[0]

        with torch.no_grad():
            for j,(latent_batch,image_batch) in enumerate(zip(self.latent_loader,self.image_loader)):

                w_ori=latent_batch[0].cuda()
                print(w_ori.shape)
                x_ori_real=image_batch[0] #real image
                x_ori, _ = self.decoder([w_ori], input_is_latent=True, randomize_noise=False, truncation=1)
                b,c,h,w = x_ori.shape

                img_name=[f"{(b_idx+j*b):0>4d}.jpg" for b_idx in range(b)]
                print(img_name)

                save_batch=torch.zeros((b,c,2*h,3*w))
                save_batch[:,:,0:h,0:w]=x_ori_real
                save_batch[:,:,0:h,w:2*w]=x_ori.cpu()

                w_pwd=self.w_pwd_list[0].repeat(b,1,1)
                w_wrong_pwd=self.w_wrong_pwd_list[0].repeat(b,1,1)
            
                w_enc=self.mapper(torch.cat([w_ori,w_pwd],dim=-1))
                x_enc, _ = self.decoder([w_enc], input_is_latent=True, randomize_noise=False, truncation=1)

                w_cat_dec=torch.cat([w_enc,w_pwd],dim=-1)
                w_dec=self.mapper(w_cat_dec)
                x_recover, _ = self.decoder([w_dec], input_is_latent=True, randomize_noise=False, truncation=1)

                w_cat_dec_wrong=torch.cat([w_enc,w_wrong_pwd],dim=-1)
                w_dec_wrong=self.mapper(w_cat_dec_wrong)
                x_wrong, _ = self.decoder([w_dec_wrong], input_is_latent=True, randomize_noise=False, truncation=1)

                save_batch[:,:,0:h,2*w:3*w]=x_enc.cpu()
                save_batch[:,:,h:2*h,w:2*w]=x_recover.cpu()
                save_batch[:,:,h:2*h,2*w:3*w]=x_wrong.cpu()
                
                for k in range(b):
                    torchvision.utils.save_image(save_batch[k],join(output_frame_path,img_name[k]),normalize=True)

    def multi_pwd_loop(self):
        output_frame_path=join(self.exp_dir,'frames')
        output_video_path=join(self.exp_dir,f'{self.exp_dir.split("/")[-1]}.mp4')
        os.makedirs(output_frame_path,exist_ok=True)
        
        with torch.no_grad():
            for j,(latent_batch,image_batch) in enumerate(zip(self.latent_loader,self.image_loader)):

                w_ori=latent_batch[0].cuda()
                x_ori_real=image_batch[0] #real image
                x_ori, _ = self.decoder([w_ori], input_is_latent=True, randomize_noise=False, truncation=1)
                b,c,h,w = x_ori.shape

                img_name=[f"{(b_idx+j*b):0>4d}.jpg" for b_idx in range(b)]
                print(img_name)

                save_batch=torch.zeros((b,c,3*h,(self.opts.pwd_num+1)*w))
                save_batch[:,:,0:h,0:w]=x_ori_real
                save_batch[:,:,h:h+h,0:w]=x_ori.cpu()

                for pwd_idx in range(len(self.w_pwd_list)):
                    w_pwd=self.w_pwd_list[pwd_idx].repeat(b,1,1)
                    w_wrong_pwd=self.w_wrong_pwd_list[pwd_idx].repeat(b,1,1)

                    w_enc=self.mapper(torch.cat([w_ori,w_pwd],dim=-1))
                    x_enc, _ = self.decoder([w_enc], input_is_latent=True, randomize_noise=False, truncation=1)

                    w_cat_dec=torch.cat([w_enc,w_pwd],dim=-1)
                    w_dec=self.mapper(w_cat_dec)
                    x_recover, _ = self.decoder([w_dec], input_is_latent=True, randomize_noise=False, truncation=1)

                    w_cat_dec_wrong=torch.cat([w_enc,w_wrong_pwd],dim=-1)
                    w_dec_wrong=self.mapper(w_cat_dec_wrong)
                    x_wrong, _ = self.decoder([w_dec_wrong], input_is_latent=True, randomize_noise=False, truncation=1)

                    save_batch[:,:,0:h,(1+pwd_idx)*w:(2+pwd_idx)*w]=x_enc.cpu()
                    save_batch[:,:,h:2*h,(1+pwd_idx)*w:(2+pwd_idx)*w]=x_recover.cpu()
                    save_batch[:,:,2*h:3*h,(1+pwd_idx)*w:(2+pwd_idx)*w]=x_wrong.cpu()
                
                for k in range(b):
                    torchvision.utils.save_image(save_batch[k],join(output_frame_path,img_name[k]),normalize=True)

if __name__=='__main__':
    opts=TestOptions().parse()
    Coach=CoachTest(opts)
    Coach.single_pwd_loop()
    #Coach.multi_pwd_loop()

