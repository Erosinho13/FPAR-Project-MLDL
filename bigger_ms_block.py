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
            nn.Linear(100*28*28, 49),
        )
        
    def forward(self, x):
        x = self.reducer(x)
        x = torch.flatten(x, 1)
        x = self.linearizer(x)
        return x
    
def msblock(**kwargs):
    return MSBlock28(**kwargs)

