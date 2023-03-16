import torch
from torch import nn
from PIL import Image
from os.path import join
import torchvision
import torchvision.transforms as trans
import sys
import torch.nn.functional as F
sys.path.append('/hd3/lidongze/animation/style_deid')
import numpy as np
from criteria.parse_related_loss.unet import unet

from argparse import Namespace


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


class BackgroundLoss(nn.Module):
    def __init__(self, opts):
        super(BackgroundLoss, self).__init__()
        print('Loading UNet for Background Loss')
        self.parsenet = unet()
        self.parsenet.load_state_dict(torch.load(opts.parsenet_weights,map_location='cpu'))
        self.parsenet.eval()
        self.bg_mask_l2_loss = torch.nn.MSELoss(size_average='mean')
        self.shrink = torch.nn.AdaptiveAvgPool2d((512, 512))
        #self.magnify = torch.nn.AdaptiveAvgPool2d((1024, 1024))
        self.magnify = torch.nn.AdaptiveAvgPool2d((256, 256))
         

    def gen_bg_mask(self, input_image):
        labels_predict = self.parsenet(self.shrink(input_image)).detach()
        #mask_512 = (torch.unsqueeze(torch.max(labels_predict, 1)[1], 1)!=13).float()
        mask_512 = (torch.unsqueeze(torch.max(labels_predict, 1)[1], 1)!=0).float() #0 for background
        #mask_1024 = self.magnify(mask_512)
        mask_256 = self.magnify(mask_512)
        return mask_256

    def forward(self, x, x_hat):
        x_bg_mask = self.gen_bg_mask(x)
        x_hat_bg_mask = self.gen_bg_mask(x_hat)
        bg_mask = ((x_bg_mask+x_hat_bg_mask)==2).float()
        loss = self.bg_mask_l2_loss(x * bg_mask, x_hat * bg_mask)
        return loss


class ParseLoss(nn.Module):
    #Parse loss for eyes and years
    def __init__(self, opts):
        super(ParseLoss, self).__init__()
        print('Loading UNet for Background Loss')
        self.parsenet = unet()
        self.parsenet.load_state_dict(torch.load(opts.parsenet_weights,map_location='cpu'))
        self.parsenet.eval()
        #self.bg_mask_l2_loss = torch.nn.MSELoss(size_average='mean')
        self.shrink = torch.nn.AdaptiveAvgPool2d((512, 512))
        #self.magnify = torch.nn.AdaptiveAvgPool2d((1024, 1024))
        self.magnify = torch.nn.AdaptiveAvgPool2d((opts.stylegan_size, opts.stylegan_size))
        self.select_list=[2,4,5,10,11,12]

    def predict(self,x):
        x_predict = self.magnify(self.parsenet(self.shrink(x)))
        select_mask=torch.zeros_like(x_predict)
        select_mask[:,self.select_list,...]=1
        return select_mask*x_predict


    def gen_mask(self, input_image):
        with torch.no_grad():
            labels_predict = self.parsenet(self.shrink(input_image)).detach()
            b,c,h,w=labels_predict.shape
            mask_final = torch.zeros((b,1,h,w)).to(labels_predict.device)
            for i in range(len(self.select_list)):
                mask_final = torch.logical_or(mask_final, (torch.unsqueeze(torch.max(labels_predict, 1)[1], 1)==self.select_list[i]))
            #mask final: (b,1,h,w)
            mask_final = mask_final.float()
            mask_final = self.magnify(mask_final)
            return mask_final

    def forward(self, x, x_hat):
        x_predict = self.predict(x)
        x_hat_predict = self.predict(x_hat)
        loss = F.mse_loss(x_predict, x_hat_predict)
        return loss

if __name__=='__main__':
    torch.set_printoptions(threshold=np.inf)

    image_size=256
    image_path="/hd3/lidongze/animation/STIT/results/training_results/obama/inversion/0000.jpg"
    result_path="/hd3/lidongze/animation/style_deid/test_result/22-6-10-test-parse"
    img=Image.open(image_path)
    preprocess=trans.Compose([trans.Resize(image_size),trans.ToTensor()])
    img=preprocess(img).cuda().unsqueeze(0)
    #img=img.repeat(2,1,1,1)
    #print(img.shape)

    


    opts={'parsenet_weights':'/hd3/lidongze/animation/style_deid/pretrained_models/parsenet.pth'}
    opts= Namespace(**opts)
    ploss=ParseLoss(opts).cuda()

    print(ploss(img,img))

    # plain,predict=ploss.predict(img)
    # print(plain.shape)
    # print(plain)
    # print(predict.shape)
    # print(predict)

    # labels_predict=ploss.parsenet(ploss.shrink(img)).detach()
    # print(torch.max(labels_predict, 1)[1].shape)



    # mask=ploss.gen_mask(img)
    # print(mask.shape)
    # torchvision.utils.save_image(torch.cat([img,mask.repeat(1,3,1,1)],dim=0),join(result_path,'test_parse.jpg'),normalize=True)
    

    #torch.unsqueeze(torch.max(labels_predict, 1)[1], 1)!=13