import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import sys
sys.path.append("/hd3/lidongze/animation/style_deid")
from models.encoders.psp_encoders import GradualStyleEncoder,Encoder4Editing,BackboneEncoderUsingLastLayerIntoW
from models.psp import get_keys,pSp
from argparse import Namespace
import numpy as np

#wnorm loss from psp
class WNormLoss(torch.nn.Module):

	def __init__(self, start_from_latent_avg=True):
		super(WNormLoss, self).__init__()
		self.start_from_latent_avg = start_from_latent_avg

	def forward(self, latent, latent_avg=None):
		if self.start_from_latent_avg:
			latent = latent - latent_avg
		return torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]

#latent editing alignment 
class LatentAlignLoss(torch.nn.Module):

    def __init__(self, opts,boundary=None):
        super(LatentAlignLoss, self).__init__()

        self.opts=opts
        if boundary is None:
            self.boundary = torch.load(opts.boundary_weights,map_location='cpu')
        else:
            self.boundary = boundary
        self.boundary= self.boundary.to('cuda')


    def forward(self, w1, w2): 
        return F.l1_loss((w2*self.boundary).sum(dim=-1,keepdim=True),(w1*self.boundary).sum(dim=-1,keepdim=True))
    
