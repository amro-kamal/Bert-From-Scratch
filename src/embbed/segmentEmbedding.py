import torch.nn as nn

class segemntEmbedding(nn.Module):
    def __init__(self , num_segments , d):
        super().__init__()
        self.seg_embd = nn.Embedding(2,d)
    def forward(self , pos):
        return self.seg_embd(pos)