import torch
import torch.nn as nn

class MSBlock28(nn.Module):    
    def __init__(self):
        super(MSBlock28, self).__init__()
        
        self.reducer = nn.Sequential(
            nn.Conv2d(128, 100, kernel_size = 1),
            nn.ReLU(inplace = True)
        )
        self.linearizer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(100*28*28, 28*28),
        )
        
    def forward(self, x):
        x = self.reducer(x)
        x = torch.flatten(x, 1)
        x = self.linearizer(x)
        return x


class MSBlock14(nn.Module):    
    def __init__(self):
        super(MSBlock14, self).__init__()
        
        self.reducer = nn.Sequential(
            nn.Conv2d(256, 100, kernel_size = 1),
            nn.ReLU(inplace = True)
        )
        self.linearizer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(100*14*14, 14*14),
        )
        
    def forward(self, x):
        x = self.reducer(x)
        x = torch.flatten(x, 1)
        x = self.linearizer(x)
        return x

    
def msblock(dim=28, **kwargs):
    if dim == 28:
        return MSBlock28(**kwargs)
    if dim == 14:
        return MSBlock14(**kwargs)

