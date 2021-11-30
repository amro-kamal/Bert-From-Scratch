from .selfAttention import selfAttention
import torch
import torch.nn as nn
import torch.Functional as Fu

class multiHeadedAttention(nn.Module):
    '''
    input shape (b X T x d)
    If the hidden size is d=768 we will divide it between the 8 heads , each 768/8
    output shape (b x T x d)
    '''
    def __init__(self, num_heads=8 , hidden_size=768 ):
        super().__init__():
        self.num_heads = num_heads
        self.dk = hidden_size // self.num_heads
        self.q_projector ,self.W_q_projector,self.W_q_projector = nn.Linear(nn.Linear(hidden_size,hidden_size)) , nn.Linear(nn.Linear(hidden_size,hidden_size)) , nn.Linear(nn.Linear(hidden_size,hidden_size))
        self.attention = selfAttention()
        self.cat_linear = nn.Linear(hidden_size,hidden_size)

    def forward(self , hiddens , mask):
        '''
          hidden : hiddden representaion of size (batch , T , d)
        '''
        # Q of shape (b , T , heads , dk)
        Q , K , V = project_and_divide_hiddens(hidddens , self.num_heads , self.dk)
        # perform self attention for each head 
        attn_out = attention(Q,K,V , mask) # (b x h x T x dk)
        b=attn_out.shape[0]
        T=attn_out.shape[2]
        attn_out = attn_out.transpose(1,2).view(b , T, -1)  #(b x T x d=h*dk)
        return cat_linear(attn_out) #(b x T x d) ==> (b x T x d)

    def project_and_divide_hiddens(hiddens , num_head):
        '''
        function to reshape the Q , K and V vetcors from d ==> h x dk
        inputs :
          hidden : hiddden representaion of size (batch , T , d)
        outputs : 
          Q , K , V : of size (batch , h , T , dk=d/h)
        '''
        batch_size = hiddens.shape[0]
        Q=q_projector(hiddens).reshape(batch_size , -1 , num_heads , self.dk ))
        K=k_projector(hiddens).reshape(batch_size , -1 , num_heads , self.dk ))
        V=v_projector(hiddens).reshape(batch_size , -1 , num_heads , self.dk ))
    
        return Q,K,V

