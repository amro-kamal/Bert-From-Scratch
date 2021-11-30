import torch.nn as nn
from encoder import encoder
from embbding import positionalEmbedding , segemntEmbedding
from bertHead import bertHead
import copy

class Bert(nn.Module):
    def __init__(self , voc_size , maxlen ,  num_layers , num_heads=8 , d=512 , fc_hidden=4*512 , dropout =0.3 ):
        super().__init__()
        self.token_embbdeings = nn.Embedding(voc_size , d)
        self.pos_embeddings = positionalEmbedding(d , maxlen) 
        self.seg_embeddings = segemntEmbedding(2 , d)
        self.encoder = encoder(voc_size , maxlen ,  num_layers , num_heads , d , fc_hidden , dropout , )
        self.head = bertHead(voc_size , d)
        self.dropout = Dropout(dropout)

    def forward(self , input  , segments):
        '''
        input : (b x T)
        mask : (b x T)
        segments : (b x T-1)

        '''
        mask = ?
        hiddens = self.dropout( self.embbdeings(input) + self.pos_embeddings()[:,:input.shape[1],:] )
        hiddens[1:] + =  self.seg_embeddings(segments)
        hiddens = self.encoder(hiddens , mask)

        CLS , masks_preds = self.bertHead(hiddens)


        return CLS , masks_preds



