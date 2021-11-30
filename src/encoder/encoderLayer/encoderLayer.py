from posionalwiseFC import posionalwiseFC
from multiHeadedAttention import multiHeadedAttention
import torch.nn as nn
import torch

class encoderLayer(nn.Module):
    def __init__(self , num_heads , d , fc_hidden):
        super().__init__():
        self.multiheadedattention = multiHeadedAttention( num_heads=num_heads , hidden_size=d)#(b x T x d) ==> (b x T x d)
        self.layernorm1= nn.LayerNorm(d) #(b x T x d) ==> (b x T x d)
        self.layernorm2= nn.LayerNorm(d) #(b x T x d) ==> (b x T x d)
        self.fc = posionalwiseFC(d, fc_hidden) #(b x T x d)==>(b x T x d)

    def forward(self ,hiddens , mask):
        '''
        inputs:
           hiddens : (b x T x d)
        output:
           hiddens : (b x T x d) 
        '''
        hiddens= skip_connection(hiddens , self.multiHeadedAttention(hiddens , mask) , self.layernorm1)
        hiddens= skip_connection(hiddens , self.posionalwiseFC(hiddens) , self.layernorm2)

        return hiddens #(b x T x d)

    def skip_connection(self , x , fx , layernorm ) :
        assert x.shape == fx.shape , f'the two tensors in the skip connection must have the same dimensions, getting {x.shape} and  {fx.shape}'
        return layernorm(x + fx)
