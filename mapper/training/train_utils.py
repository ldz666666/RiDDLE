import os
import torch
import numpy as np
import sys
import random

class RedirectLogger(object):
    def __init__(self, fileN="record.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def aggregate_loss_dict(agg_loss_dict):
	mean_vals = {}
	for output in agg_loss_dict:
		for key in output:
			mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
	for key in mean_vals:
		if len(mean_vals[key]) > 0:
			mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
		else:
			print('{} has no value'.format(key))
			mean_vals[key] = 0
	return mean_vals

def batch_tensor_to_np_row(x):
    x_np=x #(b,3,1024,1024)
    x_np=torch.cat(list(x_np.detach().cpu()),dim=-1) #b,c,h,w -> c,h,b*w
    x_np=x_np.permute(1,2,0) #h,b*w,c
    x_np=((x_np+1)/2).clamp(0,1)
    x_np=(x_np.numpy()*255).astype(np.uint8)
    return x_np

def sample_w(g,batch_size,device='cuda'):
    z = torch.randn(batch_size, 512, device=device)
    w = g.style(z).unsqueeze(1).repeat(1,14,1)
    return w

#data augment class for latent codes
class LatentAugmenter:
    def __init__(self, attribute_list=["Smiling","Narrow_Eyes","Big_Lips"],factor_range=(-3,3), latents_num=14,device='cuda',aug_method='interfacegan'):
        self.factor_range=factor_range
        self.attribute_list=attribute_list
        self.latents_num=latents_num
        self.device=device
        direction_list=[]
        for attr_name in self.attribute_list:
            attr_tensor=torch.load(f'/hd3/lidongze/animation/style_deid/interfacegan_directions/{attr_name}_256.pt',map_location='cpu')
            #print('attr tensor shape is',attr_tensor.shape)
            direction_list.append(attr_tensor)
        self.direction_bank=torch.cat(direction_list,dim=0).unsqueeze(1).to(self.device) #(n,1,512)
        #print('direction bank shape is',self.direction_bank.shape)


    def augment(self,w):
        if len(w.shape)==3:
            b,l,c=w.shape[0],w.shape[1],w.shape[2]
            assert l==self.latents_num
        for i in range(self.direction_bank.shape[0]):
            direction_repeated=self.direction_bank[i].repeat(b,self.latents_num,1)
            weight=np.random.uniform(low=self.factor_range[0],high=self.factor_range[1],size=(b,1,1)).astype(np.float32)
            weight=torch.from_numpy(weight).to(w.device)
            #print('weight shape is',weight.shape)
            w=w+weight*direction_repeated
        return w

if __name__=="__main__":
    aug=LatentAugmenter()
    w0=torch.randn(2,14,512).cuda()
    result=aug.augment(w0)
    print(result.shape)