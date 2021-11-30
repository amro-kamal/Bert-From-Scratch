import torch
import torch.nn as nn
import torch.Functional as F
 

class linear(nn.Module):
    def __init__(self , d , fc_hidden_size , dropout=0.3):
        #import dropout from config
        super().__init__()
        self.linear1 = nn.Linear(d , fc_hidden_size)
        self.linear2 = nn.Linear(fc_hidden_size , d)
        self.dropout = nn.Dropout(dropout)

    def forward(self , x):
        x=F.relu(self.dropout(self.linear1(x)))
        x=self.linear2(x)

        return x
    
        