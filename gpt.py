# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:11:12 2023

@author: leyuan

references:
    https://github.com/karpathy/nanoGPT
"""

import math
import time
from datetime import timedelta
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F



# load data
with open('tiny_shakespeare.txt', 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")



# get all the unique characters
chars = sorted(list(set(data)))     # chars[0]: '\n',  chars[1]: ' '
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size}")


# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(string):
    '''encoder: take a string, output a list of integers'''
    return [stoi[s] for s in string]

def decode(ids):
    '''decoder: take a list of integers, output a string'''
    return ''.join(itos[i] for i in ids)


print(encode("hi there"))
print(decode(encode("hi there")))
'''
Here we tokenize the data with a character-level.
For more complicated problem, you may need a better tokenizer such as tiktoken, SentencePiece or spacy

import tiktoken
enc = tiktoken.get_encoding('gpt2')  # tokenizer used in GPT-2
print(enc.n_vocab)                   # 50257
print(enc.encode("hello world"))     # [31373, 995]
print(enc.decode(enc.encode("hello world")))
'''



# create the train, validation split
n = int(0.9 * len(data))   # number of tokens
train_data = torch.as_tensor(encode(data[:n]), dtype=torch.long)
val_data = torch.as_tensor(encode(data[n:]), dtype=torch.long)
print(f"train has {len(train_data):,} tokens")
print(f"val has {len(val_data):,} tokens")
'''
you can also consider randomly sampling 90% data as training set and the other as validation set.
we just ignore the test set here.
'''

# create data loader
context_len = 8  # context of up to 8 previous characters
batch_size = 4
  
   
def get_batch(data, context_len, batch_size):
    '''generate a batch of data of inputs x and targets y'''
    idx = torch.randint(len(data) - context_len, (batch_size, ))
    
    x = torch.stack([data[i:i+context_len] for i in idx])
    y = torch.stack([data[i+1:i+context_len+1] for i in idx])
    
    return x, y



xb, yb = get_batch(train_data, context_len, batch_size)
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

'''
Illustration of what the model really sees in training process.
''' 
for b in range(batch_size):    
    for t in range(context_len):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context}, the target is {target}.")
        

        
# ==================== build GPT ==========================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, context_len, d_model, h, dropout=0.1):
        super().__init__()
        
        assert d_model % h == 0
        self.h = h
        self.d = d_model // h
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(p=dropout)
        self.dropout = nn.Dropout(p=dropout)
        
        self.attn = None
        self.register_buffer('mask', torch.tril(torch.ones(1, 1, context_len, context_len)).type(torch.int8))
        
    def forward(self, x):
        # x shape: [batch_size, len, d_model]
        # mask shape: [1, 1, len, len]
        b, t, d_model = x.shape  
        
        # pre-attention linear projection:         
        q, k, v = self.qkv(x).split(d_model, dim=-1)
        
        # separate different heads: [b, len, h*d] ==> [b, len, h, d]
        q = q.view(b, t, self.h, self.d)
        k = k.view(b, t, self.h, self.d)
        v = v.view(b, t, self.h, self.d)
        
        # transpose for attention dot product: [b, len, h, d] ==> [b, h, len, d]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        att = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        self.attn = att
        att = self.attn_dropout(att)   # Not mentioned in the paper but practically useful
        
        out = att @ v            # [b, h, len, d]
        # transpose to move the head dimension back: [b, h, len, d] ==> [b, len, h, d]
        # combine the last two dimensions to concatenate all the heads together: [b, len, h*d]
        out = out.transpose(1, 2).contiguous().view(b, t, d_model)
        out = self.proj(out)  # [b, h, len, d]
        
        del q, k ,v
        
        return self.dropout(out)


# multi_haed_attn = MultiHeadSelfAttention(h=8, d_model=512)
# x = torch.randn(64, 10, 512)
# mask = torch.tril(torch.ones(1, 1, 10, 10)).type(torch.int8)      
# output = multi_haed_attn(x, mask)    
# print(output.shape)  # [64, 10, 512]
# print(multi_haed_attn.attn.shape)    # [64, 8, 10, 10]



class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        self.w1 = nn.Linear(d_model, 4 * d_model)  
        self.w2 = nn.Linear(4 * d_model, d_model)  
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x shape: [b, len, d_model]
        x = self.w2(F.relu(self.w1(x)))         
        
        return self.dropout(x)



class DecoderBlock(nn.Module):
    def __init__(self, context_len, d_model, h, dropout):
        super(DecoderBlock, self).__init__()
        self.dec_attn = MultiHeadSelfAttention(context_len, d_model, h, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # pre-LayerNorm
        x = x + self.dec_attn(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
                
        return x
        

class Decoder(nn.Module):
    def __init__(self, context_len, d_model, h, N, dropout):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(*[DecoderBlock(context_len, d_model, h, dropout) for _ in range(N)])  
        
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, seq):

        dec_output = self.decoder(seq)
 
        return self.layer_norm(dec_output)




class GPT(nn.Module):
    def __init__(self, vocab_size, context_len, d_model, h, N, dropout):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(context_len, d_model)   # Instead of using encoding as the original paper, embedding is used here
        self.decoder = Decoder(context_len, d_model, h, N, dropout)
        self.linear = nn.Linear(d_model, vocab_size)
        
        self.context_len = context_len
    
        # # initialize the weights    
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
    
    def forward(self, seq):
        # seq shape: [batch_size, seq_len]
        b, t = seq.shape
        token_emb = self.token_embedding(seq)  # [b, t, d_model]
        pos_emb =self.positional_embedding(torch.arange(t)) # [t, d_model]
        x = token_emb + pos_emb   # [b, t, d_model]
        
        x = self.decoder(x)  # [b, t, d_model]
        
        logits = self.linear(x)   # [b, t, vocab_size]
        
        return logits
        
        

    @torch.no_grad()
    def generate(self, seq, max_len):
        # seq is [B, t] array of indices in the current context
        for _ in range(max_len):
            # crop seq to the last 'context_len' tokens, or the positional embedding will fail
            seq_crop = seq[:, -self.context_len:]
            logits = self(seq_crop)
            # focus on only the last time step
            logits = logits[:, -1, :]   # [B, vocab_size]
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            # append sampled index to the running sequence
            seq = torch.cat([seq, idx_next], dim=-1)  # [B, t+1]
            
        return seq
        


# ==================== Training ====================================
def compute_loss(model, x, y):
    # x, y shape:[b, t]
    logits = model(x)
    b, t, vocab_size = logits.shape
    logits = logits.contiguous().view(-1, vocab_size)
    y = y.contiguous().view(-1)
    loss = F.cross_entropy(logits, y)
    
    return loss


@torch.no_grad()
def eval_loss(model, eval_iters):
    train_loss = val_loss = 0
    
    model.eval()
    for _ in range(eval_iters):
        x, y = get_batch(train_data, context_len, batch_size)
        train_loss += compute_loss(model, x, y).item()
        
        x, y = get_batch(val_data, context_len, batch_size)
        val_loss += compute_loss(model, x, y).item()        
    model.train()
    
    return train_loss / eval_iters, val_loss / eval_iters


vocab_size = len(chars)
context_len = 8  # context of up to 8 previous characters
d_model = 32     # token embedding dim
h = 4            # number of heads in self-attention   
N = 2            # number of blocks in decoder
dropout = 0.1
batch_size = 32
learning_rate = 1e-3
max_iters = 5000
eval_interval = 500  # keep frequent because we'll overfit
eval_iters = 200  


seed = 3
torch.manual_seed(seed)


gpt = GPT(vocab_size, context_len, d_model, h, N, dropout)

optimizer = optim.Adam(gpt.parameters(), lr=learning_rate)


def train():
    pass

train_losses = []
val_losses = []
start = time.time()
for step in range(max_iters):   
    # every once in a while evaluate the train and val loss
    if step % eval_interval == 0:
        train_loss, val_loss = eval_loss(gpt, eval_iters)
        # print('='*30)
        print(f"step: {step}, train loss: {train_loss:.4f}, eval loss: {val_loss:.4f}, time: {timedelta(seconds=time.time()-start)}")
        # print('='*30)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
    xb, yb = get_batch(train_data, context_len, batch_size)
    
    loss = compute_loss(gpt, xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    
plt.plot(train_losses, label='train_loss')
plt.plot(val_losses, label='val_loss')
plt.legend()


gpt.eval()
start = torch.zeros((1, 1), dtype=torch.long)
print(decode(gpt.generate(start, max_len=500)[0].tolist()))



        
        
        


