import torch
import torch.nn as nn

class MSBlock(nn.Module):
    
    def __init__(self):
        super(MSBlock, self).__init__()
        
        self.reducer = nn.Sequential(
            nn.Conv2d(512, 100, kernel_size = 1),
            nn.ReLU(inplace = True)
        )
        self.linearizer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(100*7*7, 49),
            nn.Softmax(dim = 1)
        )
        
    def forward(self, x):
        x = self.reducer(x)
        x = torch.flatten(x, 1)
        x = self.linearizer(x)
        return x
    
def msblock(**kwargs):
    return MSBlock(**kwargs)