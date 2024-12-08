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

def conv_process_1234_test(parent_features1, parent_features2, parent_features3, parent_features4, child_features1,
                      child_features2, child_features3, child_features4, child_list, domain_labels, K_PAIR, BATCH_SIZE, FEATURE_DIM,
                      GPU):
#新孩子列表，负对数

    child_labels = [x % 3 for x in domain_labels]#1 & 2
    parent_labels = [x // 3 for x in domain_labels]#0 & 3
    # print("######")
    # print(child_labels)
    # print(parent_labels)
#重组孩子性别列表
    child_labels = np.array([child_labels[i] for i in child_list])
    new_domain_labels = [(x+y).numpy().item() for x,y in zip(child_labels,parent_labels) ]
#重组孩子
    new_child_feature1 = torch.Tensor(np.array([child_features1[i].cpu().detach().numpy() for i in child_list]))
    new_child_feature1 = new_child_feature1.cuda(GPU)


    parent_features_ext1 = parent_features1.unsqueeze(0).repeat(K_PAIR, 1, 1, 1, 1)
    parent_features_ext1 = torch.transpose(parent_features_ext1, 0, 1)#假设多了一组孩子，需要复制对应组父亲匹配
    parent_features_ext1 = parent_features_ext1.contiguous().view(K_PAIR * BATCH_SIZE, FEATURE_DIM, 32, 32)#拓展

    p1 = torch.cat((parent_features1, parent_features_ext1), 0)
    c1 = torch.cat((child_features1, new_child_feature1), 0)

    new_child_feature2 = torch.Tensor(np.array([child_features2[i].cpu().detach().numpy() for i in child_list]))
    new_child_feature2 = new_child_feature2.cuda(GPU)

    parent_features_ext2 = parent_features2.unsqueeze(0).repeat(K_PAIR, 1, 1, 1, 1)

    parent_features_ext2 = torch.transpose(parent_features_ext2, 0, 1)

    parent_features_ext2 = parent_features_ext2.contiguous().view(K_PAIR * BATCH_SIZE, FEATURE_DIM, 16, 16)

    p2 = torch.cat((parent_features2, parent_features_ext2), 0)
    c2 = torch.cat((child_features2, new_child_feature2), 0)

    new_child_feature3 = torch.Tensor(np.array([child_features3[i].cpu().detach().numpy() for i in child_list]))
    new_child_feature3 = new_child_feature3.cuda(GPU)

    parent_features_ext3 = parent_features3.unsqueeze(0).repeat(K_PAIR, 1, 1, 1, 1)

    parent_features_ext3 = torch.transpose(parent_features_ext3, 0, 1)
    parent_features_ext3 = parent_features_ext3.contiguous().view(K_PAIR * BATCH_SIZE, FEATURE_DIM, 16, 16)
    p3 = torch.cat((parent_features3, parent_features_ext3), 0)
    c3 = torch.cat((child_features3, new_child_feature3), 0)

    new_child_feature4 = torch.Tensor(np.array([child_features4[i].cpu().detach().numpy() for i in child_list]))
    new_child_feature4 = new_child_feature4.cuda(GPU)

    parent_features_ext4 = parent_features4.unsqueeze(0).repeat(K_PAIR, 1, 1, 1, 1)

    parent_features_ext4 = torch.transpose(parent_features_ext4, 0, 1)
    parent_features_ext4 = parent_features_ext4.contiguous().view(K_PAIR * BATCH_SIZE, FEATURE_DIM, 14, 14)

    p4 = torch.cat((parent_features4, parent_features_ext4), 0)
    c4 = torch.cat((child_features4, new_child_feature4), 0)
    return p1, p2, p3, p4, c1, c2, c3, c4,new_domain_labels



