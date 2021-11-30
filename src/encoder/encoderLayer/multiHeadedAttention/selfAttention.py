import torch.nn as nn 
import torch
class self_attention(nn.Module):
  
    def forward(self , Q , K , V , mask=None) :
        
        '''
        input:
            Q : of shape (b , h , T , dk)
            K : of shape (b , h , T , dk)
            V : of shape (b , h , T , dk)
            mask : (b , 1 , 1 , T)
        return :
            atten : of shape(b x h x T x dk)
        '''

        d=Q.shape[-1]
        similarity =troch.matmul(Q ,K.transpose(-2,-1))  #(b x h x T x dk)@(b x h x dk x T) ==> (b x h x T x T)
        if mask:
            #mask padding tokens
            similarity = similarity.masked_fill(mask==0 , -1e10)

        attention_weights = nn.Softmax(similarity / torch.sqrt(d) , dim=-1) #(b x h x T x T)
        # attention_weights : (b , h , T , T)   @  V : (b , h , T , dk) ==> (b x h x T x dk)
        return torch.matmul(attention_weights , V)
