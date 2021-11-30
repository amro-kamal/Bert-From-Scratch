import torch.nn as nn
from encoderLayer import encoderLayer
from embbed import positionalEmbedding

class encoder(nn.Module):
    def __init__(self   ,  num_layers , num_heads , d , fc_hidden ):
        super().__init__()
        self.encoder_layers = [ encoderLayer(num_heads , d , fc_hidden) for _ in range(num_layers) ]
        self.layernorm= nn.LayerNorm(d) #(b x T x d) ==> (b x T x d)

    def forward(self , hiddens , mask):
        '''
        input :
            hiddens : (b x T x d)
        return :
            hiddens : (b x T x d)
        '''
        for layer in self.encoder_layers:
            hiddens = layer(hiddens , mask)
        return self.layernorm(hiddens)



