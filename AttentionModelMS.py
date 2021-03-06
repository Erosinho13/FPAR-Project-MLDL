import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *
from resnetMod import resnet34
from MSBlock import msblock


class AttentionModelMS(nn.Module):
    def __init__(self, num_classes = 61, mem_size = 512):
        super(AttentionModelMS, self).__init__()
        
        self.num_classes = num_classes
        self.mem_size = mem_size
        
        self.resNet = resnet34(True, True)
        self.weight_softmax = self.resNet.fc.weight
        
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        
        self.classifier = nn.Sequential(self.dropout, self.fc)
        
        self.msBlock = msblock()

        
    def forward(self, inputVariable, no_cam = False, mmaps = False):
        ms_out = []
        state = (
            Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
            Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda())
        )
        
        if not no_cam:
            for t in range(inputVariable.size(0)):

                logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
                
                if mmaps:
                    ms_out.append(self.msBlock(feature_conv))
                
                bz, nc, h, w = feature_conv.size()
                feature_conv1 = feature_conv.view(bz, nc, h*w)

                probs, idxs = logit.sort(1, True)
                class_idx = idxs[:, 0]

                cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)

                attentionMAP = F.softmax(cam.squeeze(1), dim=1)
                attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)

                attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)

                state = self.lstm_cell(attentionFeat, state)
        else:
            for t in range(inputVariable.size(0)):

                logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
                
                if mmaps:
                    ms_out.append(self.msBlock(feature_conv))
                
                state = self.lstm_cell(feature_convNBN, state)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)

        if mmaps:
            ms_out = torch.stack(ms_out)
            return feats, feats1, ms_out
        
        return feats, feats1
        

def attention_model_ms(**kwargs):
    return AttentionModelMS(**kwargs)