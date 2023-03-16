# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, normalize_type='layernorm'):
        super().__init__()
        assert normalize_type in ['layernorm','instancenorm']
        print("transformer normalize_type",normalize_type)
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
            self.norm1 = nn.InstanceNorm1d(d_model)
            self.norm2 = nn.InstanceNorm1d(d_model)
            self.norm3 = nn.InstanceNorm1d(d_model)


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
        tgt = self.norm1(tgt)
        #print('after norm tgt shape',tgt.shape)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
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


class SelfAttnDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_type='layernorm'):
        super(SelfAttnDecoderLayer,self).__init__()
        assert normalize_type in ['layernorm','instancenorm','None']
        self.normalize_type=normalize_type
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

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerMapper(nn.Module):

    def __init__(self,normalize_type='layernorm'):
        #different mappers
        super(TransformerMapper, self).__init__()
        assert normalize_type in ['layernorm','instancenorm']
        self.transformerlayer_coarse = TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=1024,normalize_type=normalize_type)
        self.transformerlayer_medium = TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=1024,normalize_type=normalize_type)
        self.transformerlayer_fine = TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=1024,normalize_type=normalize_type)
    

    def split_latent_permute(self,w):
        w_coarse = w[:, :4, :].permute(1,0,2)
        w_medium = w[:, 4:8, :].permute(1,0,2)
        w_fine = w[:, 8:, :].permute(1,0,2)
        return w_coarse, w_medium, w_fine

    def forward(self,input_tensor):
        w_ori,w_pwd=torch.chunk(input_tensor,2,dim=-1)
        w_coarse_ori, w_medium_ori, w_fine_ori= self.split_latent_permute(w_ori)
        w_coarse_pwd, w_medium_pwd, w_fine_pwd= self.split_latent_permute(w_pwd)
        
        #print("w_coarse_ori.shape",w_coarse_ori.shape)
        w_coarse_out = self.transformerlayer_coarse(w_coarse_ori,w_coarse_pwd)
        w_medium_out = self.transformerlayer_medium(w_medium_ori,w_medium_pwd)
        w_fine_out = self.transformerlayer_fine(w_fine_ori,w_fine_pwd)

        w_out= torch.cat([w_coarse_out,w_medium_out,w_fine_out],dim=0)
        #print("w_out.shape",w_out.shape)
        w_out= w_out.permute(1,0,2)

        return w_out


class TransformerMapperSeq(nn.Module):

    def __init__(self,normalize_type='layernorm'):
        #different mappers
        super(TransformerMapperSeq, self).__init__()
        assert normalize_type in ['layernorm','instancenorm']
        self.transformerlayer_coarse = TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=1024,normalize_type=normalize_type)
        self.transformerlayer_medium = TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=1024,normalize_type=normalize_type)
        self.transformerlayer_fine = TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=1024,normalize_type=normalize_type)
    

    def split_latent_permute(self,w):
        w_coarse = w[:, :4, :].permute(1,0,2)
        w_medium = w[:, 4:8, :].permute(1,0,2)
        w_fine = w[:, 8:, :].permute(1,0,2)
        return w_coarse, w_medium, w_fine

    def forward(self,input_tensor):
        w_ori,w_pwd=torch.chunk(input_tensor,2,dim=-1)
        #w_coarse_ori, w_medium_ori, w_fine_ori= self.split_latent_permute(w_ori)
        w_coarse_pwd, w_medium_pwd, w_fine_pwd= self.split_latent_permute(w_pwd)
        w_ori=w_ori.permute(1,0,2)
        # print("w_coarse_pwd.shape",w_coarse_pwd.shape)
        # print("w_ori shape",w_ori.shape)
        w_coarse_out = self.transformerlayer_coarse(w_ori,w_coarse_pwd)
        w_medium_out = self.transformerlayer_medium(w_coarse_out,w_medium_pwd)
        w_out = self.transformerlayer_fine(w_medium_out,w_fine_pwd)

        #w_out= torch.cat([w_coarse_out,w_medium_out,w_fine_out],dim=0)
        #print("w_out.shape",w_out.shape)
        w_out= w_out.permute(1,0,2)

        return w_out

class SelfAttnMapper(nn.Module):

    def __init__(self,normalize_type='layernorm'):
        #different mappers
        super(SelfAttnMapper, self).__init__()
        assert normalize_type in ['layernorm','instancenorm']
        #d model is 1024 since its input is concatenated ...
        module_list=[]
        module_list.append(SelfAttnDecoderLayer(d_model=1024, nhead=4, dim_feedforward=1024,normalize_type=normalize_type))
        module_list.append(SelfAttnDecoderLayer(d_model=1024, nhead=4, dim_feedforward=1024,normalize_type=normalize_type))
        module_list.append(nn.Linear(1024,512))
        self.model=nn.Sequential(*module_list)

    def forward(self,input_tensor):
        input_tensor=input_tensor.permute(1,0,2)
        result=self.model(input_tensor)
        result=result.permute(1,0,2)
        return result
   
if __name__=="__main__":
    #mapper=TransformerMapper(normalize_type='instancenorm').cuda()
    #SelfAttnDecoderLayer(d_model=1024, nhead=4,d_out=1024, dim_feedforward=1024,normalize_type='instancenorm')
    #mapper=SelfAttnMapper(normalize_type='instancenorm').cuda()
    mapper=TransformerMapperSeq(normalize_type='instancenorm').cuda()
    w1=torch.randn(2,14,512).cuda()
    w2=torch.randn(2,14,512).cuda()
    t=mapper(torch.cat([w1,w2],dim=-1))
    print(t.shape)