import torch
from torch import nn
from models.encoders.model_irse import Backbone
import torch.nn.functional as F

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class IDLoss(nn.Module):
    def __init__(self, opts):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace for ID Loss')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(opts.ir_se50_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.opts = opts

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count


#minimize the id cossimilarity
class RevIDLoss(nn.Module):
    def __init__(self, opts):
        super(RevIDLoss, self).__init__()
        print('Loading ResNet ArcFace for ID Loss')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(opts.ir_se50_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        if self.opts.id_cos_margin is not None:
            self.margin=self.opts.id_cos_margin
        else:
            self.margin=0
        self.opts = opts

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            #loss += diff_target
            loss += max(diff_target+1,self.margin) #minimize cos similarity, margin is 0
            count += 1

        return loss / count

# a class for calculating all kinds of id loss
class IDLossExtractor(torch.nn.Module):
    def __init__(self,opts,Backbone,requires_grad=False):
        #requirement input 256*256 cropped and normalized face image tensor
        self.opts=opts
        super(IDLossExtractor, self).__init__()
        Backbone.eval()
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        
        self.margin=self.opts.id_cos_margin if self.opts.id_cos_margin is not None  else 0.1 
        
        self.output_layer=Backbone.output_layer
        self.input_layer=Backbone.input_layer
        body = Backbone.body
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), body[x])
        for x in range(3, 7):
            self.slice2.add_module(str(x), body[x])
        for x in range(7, 21):
            self.slice3.add_module(str(x), body[x])
        for x in range(21, 24):
            self.slice4.add_module(str(x), body[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def extract_feats(self, X):
        #preprocessing
        if X.shape[-1] != 256:
            X = self.pool(X)
        h=X[:, :, 35:223, 32:220]
        h=self.face_pool(h)
        #feed into to network
        h=self.input_layer(h)
        h0=h
        h = self.slice1(h)
        h1 = h
        h = self.slice2(h)
        h2 = h
        h = self.slice3(h)
        h3 = h
        h = self.slice4(h)
        h4 = h
        h = self.output_layer(h)
        h = l2_norm(h)
        return [h0,h1,h2,h3,h4,h]

    
    def calculate_loss(self,features1,features2):

        assert len(features1)==len(features2)
        
        percept_loss,id_loss=0,0
        for i in range(len(features1)-1):
            #print(features1[i].shape,features2[i].shape)
            percept_loss+=F.l1_loss(features1[i],features2[i])
        

        rev_id_loss=torch.cosine_similarity(features1[-1],features2[-1]).clamp(min=self.margin).mean()
        id_loss=(1-torch.cosine_similarity(features1[-1],features2[-1]).clamp(min=self.margin)).mean()

        return id_loss, rev_id_loss, percept_loss


    def forward(self,img1,img2):

        features1=self.extract_feats(img1)
        features2=self.extract_feats(img2)
        
        return self.calculate_loss(features1,features2)
