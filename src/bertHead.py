import torch.nn as nn


class bertHead(nn.Module):
    def __init__(self , voc , d):
        super().__init__()
        self.mlm = nn.Linear(d,voc)
        self.cls = nn.Linear(d,2)
    def forward(self, hiddens):
        mask_pred = self.mlm(hiddens[:,1: ])
        CLS = self.cls(hiddens[:,0])
        return CLS , mask_pred