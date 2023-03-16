import torch
import torch.nn.functional as F
import sys
import sys
sys.path.append('/hd3/lidongze/animation/style_deid')
from criteria.landmark_loss.mobilefacenet import MobileFaceNet
from torchvision import transforms
from argparse import Namespace

class LandmarkLoss(torch.nn.Module):
    def __init__(self, opts):
        super(LandmarkLoss, self).__init__()
        self.model = MobileFaceNet([112, 112], 136)
        self.model.load_state_dict(torch.load(opts.landmark_encoder_weights,map_location='cpu')['state_dict'])
        self.model.eval()
        print('loaded landmark encoder')
        #self.resize = transforms.Resize(112)
        self.resize = torch.nn.AdaptiveAvgPool2d((112, 112))

    def get_raw_output(self,imgs):
        resized_images = self.resize(imgs)
        outputs, _ = self.model(resized_images)
        return outputs[:,34:] #landmarks no jaw line


    def get_landmarks(self, imgs):
        """
        without preprocess (face crop)
        :param imgs: shape: torch.Size([batch size, 3, 112, 112])
        :return:
        outputs - model results scaled to img size, shape: torch.Size([batch size, 136])
        landmarks - reshaped outputs + no jawline, shape: torch.Size([batch size, 51, 2])
        """
        resized_images = self.resize(imgs)
        outputs, _ = self.model(resized_images)

        batch_size = resized_images.shape[0]
        landmarks = torch.reshape(outputs*112, (batch_size, 68, 2))

        #return outputs*112 , landmarks[:, 17:, :]
        return  landmarks[:, 17:, :]

    def forward(self, x1, x2):
        #need to denormalize for landmark
        x1=(x1+1)/2
        x2=(x2+1)/2
        landmark1,landmark2=self.get_landmarks(x1),self.get_landmarks(x2)
        #print(landmark1.shape)
        return F.mse_loss(landmark1,landmark2)

if __name__=="__main__":
    opts={'landmark_encoder_weights':'/hd3/lidongze/animation/style_deid/pretrained_models/mobilefacenet_model_best.pth.tar'}
    opts= Namespace(**opts)
    x1,x2=torch.randn(1,3,256,256,device='cuda'),torch.randn(1,3,256,256,device='cuda')
    ldm_loss=LandmarkLoss(opts).cuda()
    ldm_loss(x1,x2)