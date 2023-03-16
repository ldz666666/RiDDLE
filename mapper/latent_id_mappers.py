import torch
from torch import nn,Tensor
from torch.nn import Module
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

import os
print(os.getcwd())
import sys
sys.path.append("/hd3/lidongze/animation/RiDDLE")
#os.chdir()
from models.stylegan2.model import EqualLinear, PixelNorm
from argparse import Namespace

import copy
from typing import Optional, List
import torch.nn.functional as F


class ModulationModule(Module):
    def __init__(self, layernum): 
        super(ModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = Linear(512, 512) #mask and w
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False) #norm for x is not learnable 
        self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512)) #modulation parameters is learnable
        self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512)) #so affine is True
        self.leakyrelu = LeakyReLU()

    def forward(self, x, embedding):
        # print('into modulation forward')
        # print(x.shape)
        x = self.fc(x) #[b,n,512]
        x = self.norm(x) #[b]	 
        gamma = self.gamma_function(embedding.float())
        beta = self.beta_function(embedding.float())
        out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out


class FeatureMappingModule(nn.Module):
    def __init__(self,input_channel,layernum=18):
        super(FeatureMappingModule,self).__init__()
        self.layerum=layernum
        self.pool=nn.AdaptiveAvgPool2d((14,14))
        self.downsample1=nn.Sequential(nn.Conv2d(input_channel,128,4,2,1),nn.BatchNorm2d(128),nn.LeakyReLU())
        self.downsample2=nn.Sequential(nn.Conv2d(128,128,4,2,1),nn.BatchNorm2d(128),nn.LeakyReLU())
        self.Linear=nn.Linear(128*3*3,512)

    def forward(self,x):
        #x can be feature map [b,c,h,w] which will be projected 
        #or id embedding [b,d] which will do nothing
        if len(x.shape)==2:
            return x #id embedding instead of feature map, skip
        x=self.pool(x)
        x=self.downsample1(x)
        x=self.downsample2(x)
        x=x.view(x.shape[0],-1)
        x=self.Linear(x)
        return x


class ModulationModuleWithFeature(Module):
    def __init__(self, layernum, feature_input_channel=None): 
        super(ModulationModuleWithFeature, self).__init__()
        self.layernum = layernum
        self.fc = Linear(512, 512) #mask and w
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False) #norm for x is not learnable 
        self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512)) #modulation parameters is learnable
        self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512)) #so affine is True
        self.leakyrelu = LeakyReLU()
        if feature_input_channel is not None:
            self.feature_mapping = FeatureMappingModule(feature_input_channel)
        else:
            self.feature_mapping = None

    def forward(self, x, embedding):
        # print('into modulation forward')
        # print(x.shape)
        # embedding can be a feature map: [b,c,h,w]
        x = self.fc(x) #[b,n,512]
        x = self.norm(x) #[b]	 
        if self.feature_mapping is not None: #input is a [b,512]
            #print('into feature mapping module')
            embedding=self.feature_mapping(embedding)
        embedding=embedding.unsqueeze(1).repeat(1,self.layernum,1) #reshape to 1,18,512
        gamma = self.gamma_function(embedding.float())
        beta = self.beta_function(embedding.float())
        out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out


class LatentIDMapper(Module):
    def __init__(self, opts, layernum=18):
        super(LatentIDMapper, self).__init__()
        self.opts = opts
        self.layernum = layernum
        self.pixelnorm = PixelNorm()


        module_list=[]
        if self.opts.multi_level_feature_injection:
            feature_input_channel_list=[64,64,128,256,512]
            for i in range(5):
                module_list.append(ModulationModuleWithFeature(self.layernum,feature_input_channel_list[i]))
            module_list.append(ModulationModuleWithFeature(self.layernum))
        else:
            for i in range(6):
                module_list.append(ModulationModule(self.layernum))
        self.modulation_module_list = nn.ModuleList(module_list)
    
    #input: x: [b,n,512] n is number of styles
    #embedding can be w [b,n,512] or id embeddings [b,512]
    def forward(self, x, embedding): 
        #pixel norm style
        # print('into subid mapper')
        # print("x.shape",x.shape)
        # print("embedding.shape",embedding.shape)
        x = self.pixelnorm(x) #x [b,n,512]
        if isinstance(embedding,torch.Tensor) and len(embedding.shape)==2:
            embedding=[embedding.clone() for i in range(6)]
        for i,modulation_module in enumerate(self.modulation_module_list):
        	x = modulation_module(x, embedding[i])        
        return x

class SubIDMapper(Module):
    def __init__(self, opts, layernum,feature_input_channel):
        super(SubIDMapper, self).__init__()
        self.opts = opts
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        self.feature_input_channel=feature_input_channel
        self.feature_mapping=FeatureMappingModule(self.feature_input_channel,layernum)

        module_list=[]
        for i in range(6):
            module_list.append(ModulationModule(self.layernum))
        self.modulation_module_list = nn.ModuleList(module_list)
    
    #input: x: [b,n,512], embedding:[b,n,512] n is number of styles
    def forward(self, x, embedding):
        x = self.pixelnorm(x) #x [b,n,512]
        embedding_projected=self.feature_mapping(embedding)
        embedding_projected=embedding_projected.unsqueeze(1).repeat(1,self.layernum,1) #reshape to 1,18,512
        for i,modulation_module in enumerate(self.modulation_module_list):
        	x = modulation_module(x, embedding_projected)        
        return x

class MultiMappers(Module):
    def __init__(self,opts):
        super(MultiMappers, self).__init__()
        self.opts=opts
        feature_input_channel_list=[64,64,128,256,512,1]
        self.submappers = nn.ModuleDict(
            {f"mapper_{i}" : SubIDMapper(opts, 3, feature_input_channel_list[i])
        for i in range(6)})
        

    def forward(self,x,embedding):
        input_latent_list=[]
        for start in range(0,18,3):
            input_latent_list.append(x[:,start:start+3,:])
        output_latent_list=[]
        for i in range(len(embedding)):
            temp_latent=self.submappers[f"mapper_{i}"](input_latent_list[i],embedding[i])
            #print(f'temp latent level {i} shape',temp_latent.shape)
            output_latent_list.append(temp_latent)
        concated_latent=torch.cat(output_latent_list,dim=1)
        return concated_latent



class SimpleMapper(nn.Module):
    def __init__(self,first_dim=1024):
        super(SimpleMapper,self).__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(first_dim, 2048),
            nn.LeakyReLU(negative_slope=slope),
            # nn.BatchNorm1d(n_hid, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(2048,1024),
            nn.LeakyReLU(negative_slope=slope),
            # nn.BatchNorm1d(n_hid // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=slope),
            # nn.BatchNorm1d(n_hid // 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(512, 512)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)

class SimpleMapper_morefc(nn.Module):
    def __init__(self,first_dim=1024, morefc_num=0,use_ldm=False):
        super(SimpleMapper_morefc,self).__init__()
        slope = 0.2
        self.module_list=[nn.Linear(first_dim, 2048),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(2048,1024),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=slope)]
        for i in range(morefc_num):
            self.module_list.append(nn.Linear(512, 512))
            self.module_list.append(nn.LeakyReLU(negative_slope=slope))
        self.module_list.append(nn.Linear(512, 512))
        self.model=nn.Sequential(*(self.module_list))
        
        #landmark related
        self.use_ldm=use_ldm
        if self.use_ldm:
            self.landmark_mapping = nn.Sequential(nn.Linear(102,512),nn.LeakyReLU(negative_slope=slope))

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor, ldm=None):
        if self.use_ldm and ldm is not None:
            w_ldm=self.landmark_mapping(ldm)
            #print(w_ldm.shape)
            w_ldm=w_ldm.unsqueeze(1).repeat(1,input_tensor.shape[1],1)
            #print(w_ldm.shape)
            return w_ldm+self.model(input_tensor)
        else:
            return self.model(input_tensor)

class MaskMapper(nn.Module):
    def __init__(self):
        super(MaskMapper,self).__init__()
        #a mask mapper contains three single mappers: for id, attr and mask respectively
        self.attr_mapping = SimpleMapper(first_dim=512) #attribute
        self.id_mapping = SimpleMapper(first_dim=512) #id
        self.mask_mapping = SimpleMapper() #attribute
        self.sigmoid=nn.Sigmoid()

    def forward(self, input_tensor):
        #input tensor [b,18,1024] attr [b,18,512] concatenate with id [b,18,512]
        w_attr,w_id=torch.chunk(input_tensor,2,dim=-1)
        # print(w_attr.shape)
        # print(w_id.shape)
        output_attr=self.attr_mapping(w_attr)
        output_id=self.id_mapping(w_id)
        output_mask=self.mask_mapping(input_tensor)
        output_mask=self.sigmoid(output_mask)
        return (1-output_mask)*output_attr+output_mask*output_id

class ConcateMapper(nn.Module):
    def __init__(self):
        super(ConcateMapper,self).__init__()
        #a mask mapper contains three single mappers: for id, attr and mask respectively
        self.attr_mapping = SimpleMapper(first_dim=512) #attribute
        self.id_mapping = SimpleMapper(first_dim=512) #id
        # self.mask_mapping = SimpleMapper() #attribute
        # self.sigmoid=nn.Sigmoid()

    def forward(self, input_tensor):
        #input tensor [b,18,1024] attr [b,18,512] concatenate with id [b,18,512]
        w_attr,w_id=torch.chunk(input_tensor,2,dim=-1)
        w_attr=w_attr[:,:,:]
        # print(w_attr.shape)
        # print(w_id.shape)
        output_attr=self.attr_mapping(w_attr)
        output_id=self.id_mapping(w_id)
        output_mask=self.mask_mapping(input_tensor)
        output_mask=self.sigmoid(output_mask)
        return (1-output_mask)*output_attr+output_mask*output_id


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, normalize_type='layernorm'):
        super().__init__()
        assert normalize_type in ['layernorm','instancenorm','None']
        print("transformer normalize_type",normalize_type)
        self.normalize_type=normalize_type
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if normalize_type=='layernorm':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

        elif normalize_type=='instancenorm':
            self.norm1 = nn.InstanceNorm1d(d_model,affine=True)
            self.norm2 = nn.InstanceNorm1d(d_model,affine=True)
            self.norm3 = nn.InstanceNorm1d(d_model,affine=True)


        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,):
        
        #print('in transformer block, during forwarding query pos is',query_pos)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        #print('before norm tgt shape',tgt.shape)
        if self.normalize_type != 'None':
            tgt = self.norm1(tgt)
        #print('after norm tgt shape',tgt.shape)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        if self.normalize_type  != 'None':
            tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        if self.normalize_type  != 'None':
            tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SelfAttnDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_type='layernorm'):
        super(SelfAttnDecoderLayer,self).__init__()
        assert normalize_type in ['layernorm','instancenorm']
        #print("transformer normalize_type",normalize_type)
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if normalize_type=='layernorm':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

        elif normalize_type=='instancenorm':
            self.norm1 = nn.InstanceNorm1d(d_model)
            self.norm2 = nn.InstanceNorm1d(d_model)
            self.norm3 = nn.InstanceNorm1d(d_model)


        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,):
        
        #print('in transformer block, during forwarding query pos is',query_pos)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn1(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        #print('before norm tgt shape',tgt.shape)
        if self.normalize_type is not None:
            tgt = self.norm1(tgt)
        #print('after norm tgt shape',tgt.shape)
        tgt2 = self.self_attn2(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(tgt, pos),
                                   value=self.with_pos_embed(tgt, pos), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        if self.normalize_type is not None:
            tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        if self.normalize_type is not None:
            tgt = self.norm3(tgt)
        return tgt


class TransformerMapperSplit(nn.Module):

    def __init__(self,split_list=[4,4,6],normalize_type='layernorm',add_linear=False,add_pos_embedding=False):
        #different mappers
        super(TransformerMapperSplit, self).__init__()
        assert normalize_type in ['layernorm','instancenorm','None']
        layer_list=[]
        self.split_list=split_list
        self.add_linear=add_linear
        self.add_pos_embedding=add_pos_embedding
        if self.add_linear:
            self.linear_ori=nn.Linear(512,512)
            self.linear_pwd=nn.Linear(512,512)
        if self.add_pos_embedding:
            self.pos_embedding=nn.Parameter(torch.randn(sum(self.split_list),1, 512))
        for i in range(len(split_list)):
            layer_list.append(TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=1024,normalize_type=normalize_type))
        self.layers=nn.ModuleList(layer_list)


    def split_latent(self,w):
        w_split=torch.split(w,self.split_list,dim=0)
        return w_split

    def forward(self,input_tensor):

        w_ori,w_pwd=torch.chunk(input_tensor,2,dim=-1)
        #permute first
        w_ori = w_ori.permute(1,0,2)
        w_pwd = w_pwd.permute(1,0,2)
       
        #linear
        if self.add_linear:
            w_ori = self.linear_ori(w_ori)
            w_pwd = self.linear_pwd(w_pwd)
         #pos embedding
        if self.add_pos_embedding:
            w_pos_split =self.split_latent(self.pos_embedding)
        
        w_ori_split = self.split_latent(w_ori)
        w_pwd_split = self.split_latent(w_pwd)
        
        w_out_list=[]
        for i in range(len(self.layers)):
            if self.add_pos_embedding:
                w_split_temp=w_ori_split[i]+w_pos_split[i]
            else:
                w_split_temp=w_ori_split[i]
            w_out_temp=self.layers[i](w_split_temp, w_pwd_split[i])
            w_out_list.append(w_out_temp)
            
        w_out = torch.cat(w_out_list,dim=0)
        w_out = w_out.permute(1,0,2)

        return w_out

#test code for id mapper
if __name__=="__main__":
    # torch.set_printoptions(threshold=np.inf)
    # opts={'multi_level_feature_injection':True}
    # opts=Namespace(**opts)
    # lt_mapper=LatentIDMapper(opts,18)
    # test_id=[torch.randn(1,64,112,112),torch.randn(1,64,56,56),torch.randn(1,128,28,28),torch.randn(1,256,14,14),\
    #     torch.randn(1,512,7,7),torch.randn(1,512)]
    # test_latent=torch.randn(1,18,512)
    # test_mask=torch.randn(1,18,512)
    # print(lt_mapper(test_latent,test_id).shape)

    mapper=TransformerMapperSplit().cuda()
    print(mapper)
    w1=torch.randn(2,14,512).cuda()
    w2=torch.randn(2,14,512).cuda()
    t=mapper(torch.cat([w1,w2],dim=-1))
    print(t.shape)
