from genericpath import isdir
from posixpath import pardir
import torch
from torch._C import default_generator
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
import cv2 as cv
import yaml
import time
from os.path import join
import json
from torch.functional import Tensor
# from  conv.related_net import  RelationNetwork_1234_test as relation_test
from conv.loss_func import FC as MVLoss
from conv.loss_func import loss_final

parser = argparse.ArgumentParser(description="kinship Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)#loss维度,如果使用ce就把维度换成2，triplet就换成1
parser.add_argument("-b","--batch_size",type = int, default = 32)
parser.add_argument("--config",type = str,default='./configs/configs.yml')
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-s","--select_split",type = int, default=1,help='1,2,3,4,5')   # 调参
parser.add_argument("-t","--dataset",type = str, default='KINFACE1',help='TSKINFACE or KINFACE1 or KINFACE2') # 调参
parser.add_argument("--result_save_path",type = str, default='./KINFACE1_test')
parser.add_argument("--is_save_weights",type = bool, default=False)
parser.add_argument("--addtion",type = str,help='description for this training')
parser.add_argument("--modelpath",type = str, default="./KINFACE2_model")

parser.add_argument("--loss",type = str, default= 'MV',help='triplet or CE or MV')
parser.add_argument("--is_save_log",type = bool, default= True,help='')
parser.add_argument("--log_path",type = str, default= 'log',help='./')

parser.add_argument("--pretrained",type = bool, default=True)
parser.add_argument("-d","--dataroot",type = str)
parser.add_argument("-p","--picroot",type = str)
args = parser.parse_args()

 

from data.loader_kinface import load_kinface,get_train_loader_kinface1,make_test_indices,kinface_test_dataset
configs = yaml.load(open(args.config, 'rb'), Loader=yaml.Loader)
args.dataroot = configs[args.dataset]['dataroot']
args.picroot = configs[args.dataset]['picroot']


if not os.path.exists(args.modelpath):
    os.mkdir(args.modelpath)
if not os.path.exists(args.result_save_path):
    os.mkdir(args.result_save_path)
if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)



FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
BATCH_SIZE = args.batch_size
GPU = args.gpu
SELECT_SPLIT = args.select_split




from conv.related_net_duikang_beta import CNNEncoder_1234 as CNNEncoder
from conv.related_net_duikang_beta import RelationNetwork_1234 as RelationNetwork
from conv.related_net_duikang_beta import Domain_RelationNetwork_1234_0 as Domain_RelationNetwork
from conv.related_op import conv_process_1234_test

feature_F_D,feature_F_S,feature_M_D,feature_M_S,F_S_COLLECTION,M_S_COLLECTION,F_D_COLLECTION,M_D_COLLECTION = load_kinface(args.dataset,args.dataroot)




def main():
    # step 1: init dataset


    # if args.dataset == 'KINFACE1':
    related_list = get_train_loader_kinface1(args.dataset, SELECT_SPLIT, F_S_COLLECTION, F_D_COLLECTION, M_D_COLLECTION,
                                             M_S_COLLECTION, BATCH_SIZE, args.picroot)
    train_loader = related_list[0]

    # init network
    print("init neural networks")
    feature_encoder_parents = CNNEncoder()
    feature_encoder_children = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)
    if args.loss =='MV':
        MV = MVLoss(embedding_size=RELATION_DIM)

    if args.pretrained:
        feature_encoder_parents.load_state_dict(
            torch.load('KINFACE2_model/feature_encoder_parentsKINFACE1_KINFACE1_conv1234_MV_K_PAIR_1_SPLIT_1.pth'))
        feature_encoder_children.load_state_dict(
            torch.load('KINFACE2_model/feature_encoder_childrenKINFACE1_KINFACE1_conv1234_MV_K_PAIR_1_SPLIT_1.pth'))
        relation_network.load_state_dict(
            torch.load('KINFACE2_model/relation_network_KINFACE1_KINFACE1_conv1234_MV_K_PAIR_1_SPLIT_1.pth'))
        if args.loss == 'MV':
            MV.load_state_dict(
                torch.load('KINFACE2_model/MVLossKINFACE1_KINFACE1_conv1234_MV_K_PAIR_1_SPLIT_1.pth')
            )
        # domain_relation_network.load_state_dict(torch.load('KINFACE2_model/domainKINFACE1_KINFACE1_conv1234_MV_K_PAIR_1_SPLIT_1.pth'))

    feature_encoder_parents.cuda(GPU)
    feature_encoder_children.cuda(GPU)
    relation_network.cuda(GPU)
    if args.loss == 'MV':
        MV.cuda(GPU)

    print("Testing...")
    feature_encoder_parents.eval()
    feature_encoder_children.eval()
    relation_network.eval()
    if args.loss == 'MV':
        MV.eval()

    if args.dataset == 'KINFACE1':

        def compute_accuracy(feature):
            total_rewards = 0
            counter = 0
            correct_list = []
            incorrect_list = []

            if args.dataset == 'KINFACE1':
                indices = make_test_indices(feature, SELECT_SPLIT)
                feature = feature[indices]
            # print(feature.shape)

            test_root = args.picroot

            TEST_DATASET = kinface_test_dataset(feature, test_root)
            test_loader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=True)
            for features, test_labels, feature_pair in test_loader:
                batch_size = test_labels.shape[0]

                parent_set, child_set = features.split(split_size=1, dim=1)
                # print(sample_SD.reshape(32,6272).shape)
                # print(support_set.size())
                parent_feature = Variable(parent_set.view(-1, 3, 64, 64)).cuda(GPU).float()  # 32*1024
                child_feature = Variable(child_set.view(-1, 3, 64, 64)).cuda(GPU)  # k*312


                parent_feature1, parent_feature2, parent_feature3, parent_feature4 = feature_encoder_parents(
                    parent_feature)
                child_feature1, child_feature2, child_feature3, child_feature4 = feature_encoder_children(
                    child_feature)

                # relations,x = relation_network(relation_pairs1,relation_pairs2,relation_pairs3,relation_pairs4)

                relations = relation_network(parent_feature1, parent_feature2, parent_feature3, parent_feature4,
                                             child_feature1, child_feature2, child_feature3, child_feature4)
                if args.loss == "MV":
                    weight = MV.weight
                    relations = torch.mm(relations, weight)

                if args.loss == 'MV' or args.loss == "CE":
                    predict_labels = relations.argmax(1).long()
                if args.loss == "triplet":
                    predict_labels = torch.gt(relations.data, 0.5).long()

                rewards = [1 if predict_labels[j] == test_labels[j].cuda(GPU) else 0 for j in
                           range(batch_size)]

                for idex in range(batch_size):
                    if predict_labels[idex] == test_labels[idex].cuda(GPU):
                        correct_list.append(
                            str(str(feature_pair[0][idex]) + '   ' + str(relations.data[idex])))
                    else:
                        incorrect_list.append(
                            str(str(feature_pair[0][idex]) + '   ' + str(relations.data[idex])))

                total_rewards += np.sum(rewards)
                counter += batch_size
                print("counter,total_rewards is", counter, total_rewards)
            accuracy = total_rewards / 1.0 / counter

            return accuracy, correct_list, incorrect_list

        # print("father -- son")
        kinship_accuracy_for_FS, correct_fs, incorrect_fs = compute_accuracy(feature_F_S)
        # print("father -- daughter")
        kinship_accuracy_for_FD, correct_fd, incorrect_fd = compute_accuracy(feature_F_D)
        # print("mother -- son")
        kinship_accuracy_for_MS, correct_ms, incorrect_ms = compute_accuracy(feature_M_S)
        # print("mother -- daughter")
        kinship_accuracy_for_MD, correct_md, incorrect_md = compute_accuracy(feature_M_D)

        Mean = (
                       kinship_accuracy_for_FS + kinship_accuracy_for_FD + kinship_accuracy_for_MS + kinship_accuracy_for_MD) / 4.0
        print('Mean:', Mean)
        print('kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
            kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
            kinship_accuracy_for_MD))


if __name__ == '__main__':
    main()