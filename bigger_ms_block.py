import torch
import torch.nn as nn

class MSBlock28(nn.Module):    
    def __init__(self, reg=True):
        super(MSBlock28, self).__init__()
        exit_neur = 28*28
        if not reg:
            exit_neur = 28*28*2
        self.reducer = nn.Sequential(
            nn.Conv2d(128, 100, kernel_size = 1),
            nn.ReLU(inplace = True)
        )
        self.linearizer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(100*28*28, exit_neur),
        )
        
    def forward(self, x):
        x = self.reducer(x)
        x = torch.flatten(x, 1)
        x = self.linearizer(x)
        return x


class MSBlock14(nn.Module):    
    def __init__(self, reg=True):
        super(MSBlock14, self).__init__()
        exit_neur = 14*14
        if not reg:
            exit_neur = 14*14*2
        self.reducer = nn.Sequential(
            nn.Conv2d(256, 100, kernel_size = 1),
            nn.ReLU(inplace = True)
        )
        self.linearizer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(100*14*14, exit_neur),
        )
        
    def forward(self, x):
        x = self.reducer(x)
        x = torch.flatten(x, 1)
        x = self.linearizer(x)
        return x

    
def msblock(dim=28, reg=True,**kwargs):
    if dim == 28:
        return MSBlock28(reg, **kwargs)
    if dim == 14:
        return MSBlock14(reg, **kwargs)

