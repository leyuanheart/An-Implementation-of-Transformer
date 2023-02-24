# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:21:05 2023

@author: leyuan
"""

import numpy as np
import copy

import torch
from torch import nn
from torch.nn import functional as F



class ScaleDotProductAttention(nn.Module):
    "Scale Dot-Production Attention."
    def __init__(self, d_k, dropout=0.1):
        super(ScaleDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        # query shape: [batch_size, head_num, seq_len, d_k]
        # print(mask.shape, query.shape, key.shape)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)   # 最后两维做矩阵乘法
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)     # 论文中没提，但是源代码中有加
        
        return torch.matmul(attn, value), attn



class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, dropout=0.1):
        '''
        Parameters
        ----------
        h : number of heads.
        d_model : input embedding dim.
        d_k : key and query dim.
        d_v : value dim.
        dropout : dropout rate.
        '''
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % h == 0
        assert d_model // h == d_k
        self.h = h
        # self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model, h*d_k)
        self.w_ks = nn.Linear(d_model, h*d_k)      # 如果按照论文里面的设置, d_v=d_k, 那么其实这4个layer都是nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, h*d_v)        
        self.fc = nn.Linear(h*d_v, d_model)
        
        
        self.attention = ScaleDotProductAttention(d_k, dropout)
        self.attn = None   # multi-head attention weights
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        # query, key, value shape type: [batch_size, len, d_model]
        d_k, d_v, h = self.d_k, self.d_v, self.h
        batch_size, len_q, len_k, len_v = query.shape[0], query.shape[1], key.shape[1], value.shape[1]      
        
        residual = query  
        
        # pre-attention linear projection:         
        # separate different heads: [b, len, h*d] ==> [b, len, h, d]
        q = self.w_qs(query).view(batch_size, len_q, h, d_k)
        k = self.w_ks(key).view(batch_size, len_k, h, d_k)     # len的部分用-1也是可以的
        v = self.w_vs(value).view(batch_size, len_v, h, d_v)
        
        # transpose for attention dot product: [b, len, h, d] ==> [b, h, len, d]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        
        # apply attention on all the projected vectors in batch.
        if mask is not None:
            mask = mask.unsqueeze(dim=1) # for head axis broadcasting
        
        out, self.attn = self.attention(q, k, v, mask=mask)
        # transpose to move the head dimension back: [b, h, len, d] ==> [b, len, h, d]
        # combine the last two dimensions to concatenate all the heads together: [b, len, h*d]
        out = out.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        
        out = self.fc(out)

        out = self.dropout(out)
        out += residual                     # Dropout + Residual Connection + Layer Norm
        out = self.layer_norm(out)
        
        del q, k ,v
        
        return out



class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.w1 = nn.Linear(d_model, hidden_dim)  # position-wise
        self.w2 = nn.Linear(hidden_dim, d_model)  # position-wise
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.w2(F.relu(self.w1(x)))
        
        x = self.dropout(x)
        x += residual                # Dropout + Residual Connection + Layer Norm
        x = self.layer_norm(x)
        
        return x



class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        # self.vocab_size = vocab_size
        # self.d_model = d_model
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
               
    def forward(self, x):
        
        return self.embedding(x)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()  
        # self.d_model = d_model
          
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim=0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, embedding):
        "Output token embedding + positional encoding."
        # embedding shape: [batch_size, len, d_model]
        x = embedding + self.pe[:, :embedding.size(1), :].requires_grad_(False)
        
        return x



class EncoderBlock(nn.Module):
    def __init__(self, self_attn, feed_forward):
        super(EncoderBlock, self).__init__()
        self.enc_attn = self_attn
        self.feed_forward = feed_forward    
        
    def forward(self, x, mask=None):
        x = self.enc_attn(x, x, x, mask)   # padding mask
        x = self.feed_forward(x)
                
        return x
        

class Encoder(nn.Module):
    def __init__(self, d_model, src_embedding, positional_encoder, block, N, dropout=0.1, scale_emb=True):
        super(Encoder, self).__init__()
        
        self.src_embedding = src_embedding
        self.positional_encoder = positional_encoder
        self.encoder = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
        
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.scale_emb = scale_emb


    def forward(self, src_seq, src_mask=None):
        enc_slf_attn_list = []

        enc_output = self.src_embedding(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model**0.5
            # 乘sqrt(d_model)是为了让embeddings vector相对大一些，避免因加入positional info而丢失了embedding的info (也就是说位置信息相对来说没有那么重要)
        enc_output = self.dropout(self.positional_encoder(enc_output))
        enc_output = self.layer_norm(enc_output)   

        for block in self.encoder:
            enc_output = block(enc_output, mask=src_mask)
            enc_slf_attn_list += [block.enc_attn.attn] 
 
        return enc_output, enc_slf_attn_list




class DecoderBlock(nn.Module):
    def __init__(self, self_attn, enc_dec_attn, feed_forward):
        super(DecoderBlock, self).__init__()
        self.dec_attn = self_attn
        self.enc_dec_attn = enc_dec_attn  # attention to connect encoder and decoder
        self.feed_forward = feed_forward
        
        
    def forward(self, dec_input, enc_output, self_mask=None, enc_dec_mask=None):
        x = self.dec_attn(dec_input, dec_input, dec_input, self_mask)   # padding mask + sequence mask       
        x = self.enc_dec_attn(x, enc_output, enc_output, enc_dec_mask)    # padding mask if needed, 且默认encoder输出的结果既作为key, 也作为value     
        x = self.feed_forward(x)
        
        return x



class Decoder(nn.Module):
    def __init__(self, d_model, tgt_embedding, positional_encoder, block, N, dropout=0.1, scale_emb=True):
        super(Decoder, self).__init__()
        
        self.tgt_embedding = tgt_embedding
        self.positional_encoder = positional_encoder
        self.decoder = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])        
        
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.scale_emb = scale_emb

    def forward(self, tgt_seq, enc_output, src_mask=None, tgt_mask=None):
        dec_slf_attn_list, enc_dec_attn_list = [], []
        
        dec_output = self.tgt_embedding(tgt_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.positional_encoder(dec_output))
        dec_output = self.layer_norm(dec_output)

        for block in self.decoder:
            dec_output = block(dec_output, enc_output, self_mask=tgt_mask, enc_dec_mask=src_mask)
            dec_slf_attn_list += [block.dec_attn.attn] 
            enc_dec_attn_list += [block.enc_dec_attn.attn] 

        
        return dec_output, dec_slf_attn_list, enc_dec_attn_list




class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # x shape: [batch_size, len, d_model]
        logit = self.proj(x)
        return logit




class Transformer(nn.Module):
    def __init__(self, encoder, decoder, generator):
        super(Transformer, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        
        output = self.decode(tgt, self.encode(src, src_mask)[0], src_mask, tgt_mask)[0]  # 不包括attention权重
        
        return output
    
    
    def encode(self, src, src_mask=None):
        
        return self.encoder(src, src_mask)    # enc_output, enc_attn
    
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        
        return self.decoder(tgt, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)  # dec_output, dec_attn, enc_dec_attn




def make_model(src_vocab_size, tgt_vocab_size, 
               d_model=512, d_k=64, d_v=64, hidden_dim=2048, 
               N=6, h=8, max_len=500, dropout=0.1, scale_emb=True):
    "Helper: Construct a transformer from hyperparameters."
    
    c = copy.deepcopy
    embedding = Embeddings(src_vocab_size, d_model)
    position_encoder = PositionalEncoding(d_model, max_len)
    attn = MultiHeadAttention(h, d_model, d_k, d_v, dropout)
    ffn = PositionWiseFeedForward(d_model, hidden_dim, dropout)
    generator = Generator(d_model, tgt_vocab_size)

    encoder_block = EncoderBlock(c(attn), c(ffn))
    encoder = Encoder(d_model, c(embedding), c(position_encoder), encoder_block, 6, dropout, scale_emb)

    decoder_block = DecoderBlock(c(attn), c(attn), c(ffn))    
    decoder = Decoder(d_model, c(embedding), c(position_encoder), decoder_block, 6, dropout, scale_emb)

    transformer = Transformer(encoder, decoder, generator)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer



def padding_mask(seq, pad):
    "'pad' is the token to fill the blank positions of a seq. 在使用的时候seq会被编码成数字，pad也会变成token的idx。"
    return (seq != pad).unsqueeze(-2)    # 在最后一维前面插入


def sequence_mask(seq_len):
    "For masking out the subsequent info."
    attn_shape = (1, seq_len, seq_len)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def scheduler(step, d_model, factor, warmup):
    "learning rate scheduler."
    if step == 0:  # we have to default the step to 1 for LambdaLR function to avoid zero raising to negative power.
        step = 1
    return factor * (
        d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )




class LabelSmoothingKL(nn.Module):
    "Implement label smoothing."
    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        "padding_idx是指用来padding的字符在vocab里的位置."
        super(LabelSmoothingKL, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, x, target):
        # x shape: [batch_size*seq_len, vocab_size]
        # target shape: [batch_size*seq_len]
        assert x.size(1) == self.vocab_size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # 本来是-1的，多减一个是因为还有一个padding字符。
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0    # 把padding字符所在的那一列给概率给赋成0
        mask = torch.nonzero(target.data == self.padding_idx)  # 得到seq中padding字符所在位置
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)  # padding mask的行也赋值成0
        self.true_dist = true_dist
        return nn.KLDivLoss(reduction='sum')(x, true_dist.clone().detach())  # x是log_prob
    

'''
如果是用CrossEntropy Loss, 那么pytorch中的CrossEntropyLoss类自带了label smoothing和ignore_index (用来屏蔽padding的字符)
LabelSmoothingCrossEntropy = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='sum', label_smoothing=0.0)    
x是logit
'''    