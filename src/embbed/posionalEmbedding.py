import torch.nn as nn


class positionsalEncoding(nn.Module):
    def __init__(self, d , maxlen=5000):
        super().__init__()
        
        # Compute the positionsal encodings once in log space.
        pe = torch.zeros(maxlen, d) 
        # print(f'pe shape {pe.shape}') # (T x d)
        positions = torch.arange(0, maxlen).unsqueeze(1) 
        # print(f'positions shape {positions.shape}') # (T x 1)
        div_term = torch.exp(torch.arange(0, d, 2) *
                             -(math.log(10000.0) / d))
        # print(f'div term shape {div_term.shape}') #256 = d/2
        #fill even indices with sin
        pe[:, 0::2] = torch.sin(positions * div_term)
        # print(f'sin shape {torch.sin(positions * div_term).shape}') #(T x 1) * (d/2) = (T x d/2)
        #fill odd indices with cos 
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.pe = pe.unsqueeze(0)
        # print(f'pe shape {pe.shape}') #(1 x maxlen x d)
        
    def forward(self, x):
        '''
        pe : (1 x maxlen x d)  
        '''
        # print(f' x shape {x.shape} result shape {(x + self.pe[:, :x.size(1)] ).shape}')
        return self.pe  #(1 x maxlen x d)