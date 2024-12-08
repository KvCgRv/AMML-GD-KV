import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
import math
import argparse
import random
import os
from sklearn.metrics import accuracy_score
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from conv.ema_attention import EMA

class CNNEncoder_1234(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder_1234, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.LeakyReLU(),

                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                       nn.LeakyReLU(),

                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.LeakyReLU()

            )

        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.LeakyReLU()
        )


    def forward(self,x):
        out1 = self.layer1(x)
        #out1 = self.layer5(out1) + out1
        out2 = self.layer2(out1)
        #out2 = self.layer5(out2) + out2
        out3 = self.layer3(out2)
        #out3 = self.layer5(out3) + out3
        out4 = self.layer4(out3)
        # p2 = F.pad(out2, [8, 8, 8, 8])
        # p3 = F.pad(out3, [8, 8, 8, 8])
        # p4 = F.pad(out4, [9, 9, 9, 9])
        # output = torch.cat((out1, p2, p3, p4), 1)
        #
        # return out1,out2,out3,out4,output # 64
        return out1, out2, out3, out4,



# 第四版：修改损失函数，将两个特征图embedding，使得亲属样本的距离更近
class RelationNetwork_1234(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork_1234, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(512,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.LeakyReLU(),

                        nn.MaxPool2d(2))  # 32*32  ---> 15*15

        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.LeakyReLU(),

                        nn.MaxPool2d(2))    # 15*15 ---> 6*6

        self.layer3_test = nn.Sequential(
                    nn.Conv2d(64,128,kernel_size=1,padding=0),
                    nn.BatchNorm2d(128,momentum=1,affine=True),
                    nn.LeakyReLU(),

                    nn.AvgPool2d(10)

        )
        self.batch_norm_layer = torch.nn.BatchNorm2d(num_features=1)
        self.fc1 = nn.Linear(6*6*input_size,6*input_size)
        self.fc2 = nn.Linear(6*input_size,input_size)
        self.fc2_test = nn.Linear(input_size,hidden_size)



    def forward(self,p1,p2,p3,p4,c1,c2,c3,c4):
        p2 = F.pad(p2, [8, 8, 8, 8])
        p3 = F.pad(p3, [8, 8, 8, 8])
        p4 = F.pad(p4, [9, 9, 9, 9])
        c2 = F.pad(c2, [8, 8, 8, 8])
        c3 = F.pad(c3, [8, 8, 8, 8])
        c4 = F.pad(c4, [9, 9, 9, 9])
        x = torch.cat((p1,c1,p2,c2,p3,c3,p4,c4),1)
        y1 = self.layer1(x)
        #y1 = self.layer5(y1)
        y2 = self.layer2(y1)
        y2 = y2.view(y2.size()[0],-1)
        # 考虑是否需要损失函数
        # output = F.relu(self.fc1(y2))  # embedding为一个8维向量
        # output = F.relu(self.fc2(output))
        output = self.fc1(y2)  # embedding为一个8维向量
        output = self.fc2(output)
        output = F.sigmoid(self.fc2_test(output))
        #output = F.relu(self.fc2_test(output))
        return output

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class Domain_RelationNetwork_1234_0(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(Domain_RelationNetwork_1234_0, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(512,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.LeakyReLU(),
                        nn.MaxPool2d(2))  # 32*32  ---> 15*15

        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.LeakyReLU(),
                        nn.MaxPool2d(2))    # 15*15 ---> 6*6

        self.layer3_test = nn.Sequential(
                    nn.Conv2d(64,128,kernel_size=1,padding=0),
                    nn.BatchNorm2d(128,momentum=1,affine=True),
                    nn.LeakyReLU(),
                    nn.AvgPool2d(10)

        )
        self.layer5 = EMA(64).cuda()
        self.batch_norm_layer = torch.nn.BatchNorm2d(num_features=1)
        self.fc1 = nn.Linear(6 * 6 * input_size, 6 * input_size)
        self.fc2 = nn.Linear(6 * input_size, input_size)
        self.fc2_test = nn.Linear(input_size, 4)#4就是ace，1就是度量loss


    def forward(self, p1, p2, p3, p4, c1, c2, c3, c4, alpha=1.0):
        p2 = F.pad(p2, [8, 8, 8, 8])
        p3 = F.pad(p3, [8, 8, 8, 8])
        p4 = F.pad(p4, [9, 9, 9, 9])
        c2 = F.pad(c2, [8, 8, 8, 8])
        c3 = F.pad(c3, [8, 8, 8, 8])
        c4 = F.pad(c4, [9, 9, 9, 9])
        x = torch.cat((p1, c1, p2, c2, p3, c3, p4, c4), 1)
        y1 = self.layer1(x)

        # Gradient Reversal Layer
        y1 = GradientReversalLayer.apply(y1, alpha)

        y2 = self.layer2(y1)
        y2 = y2.view(y2.size()[0], -1)
        # output = F.relu(self.fc1(y2))
        # output = F.relu(self.fc2(output))
        output = self.fc1(y2)
        output = self.fc2(output)
        output = self.fc2_test(output)
        return output