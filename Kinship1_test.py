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
parser.add_argument("-r","--relation_dim",type = int, default = 12)#loss维度,如果使用ce就把维度换成2，triplet就换成1
parser.add_argument("-b","--batch_size",type = int, default = 32)
parser.add_argument("-e","--episode",type = int, default= 1000)   # 调参 not 200 else 1000
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)  # 调参,mvloss就是0.00005，ce为0.0005,triplet就是?
parser.add_argument("--config",type = str,default='./configs/configs.yml')
parser.add_argument("-g","--gpu",type=int, default=0)

parser.add_argument("-s","--select_split",type = int, default=1,help='1,2,3,4,5')   # 调参
parser.add_argument("-k","--k_neg_pair",type = int, default=1)   # 调参


# Hyper Parameters

parser.add_argument("-t","--dataset",type = str, default='KINFACE1',help='TSKINFACE or KINFACE1 or KINFACE2') # 调参
parser.add_argument("--conv",type = str, default='conv1234',help='conv14 or conv1234')
parser.add_argument("--eval_each_episode",type = int,default=5)
# parser.add_argument("--times",type = int,default=5,help='repetition times for Parameters test')
parser.add_argument("--result_save_path",type = str, default='./KINFACE2_test')
parser.add_argument("--is_save_weights",type = bool, default=True)
parser.add_argument("--addtion",type = str,help='description for this training')
parser.add_argument("--modelpath",type = str, default="./KINFACE2_model")

parser.add_argument("--discriminator",type = bool, default = True ,help='Bool')#新增可调参数
parser.add_argument("--discriminator_loss",type = str, default='ACE',help='ACE or CE')


parser.add_argument("--loss",type = str, default= 'MV',help='triplet or CE or MV')
parser.add_argument("--is_save_log",type = bool, default= True,help='')
parser.add_argument("--log_path",type = str, default= 'log',help='./')
parser.add_argument("--pretrained",type = bool, default=False)
# parser.add_argument("--feature_encoder",type = str, default="./1/feature_encoder_0TSKINFACE_conv1234_awk30_K_PAIR_4_SPLIT_1_a_0.5_m_0.5.pkl")
# parser.add_argument("--relation_network",type = str, default="./1/relation_network_0TSKINFACE_conv1234_awk30_K_PAIR_4_SPLIT_1_a_0.5_m_0.5.pkl")
parser.add_argument("-d","--dataroot",type = str)
parser.add_argument("-p","--picroot",type = str)
args = parser.parse_args()


args.addtion = "{}_{}_{}".format(args.dataset,args.conv,args.loss)



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
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
SELECT_SPLIT = args.select_split
K_PAIR = args.k_neg_pair



if args.conv == 'conv1234':
    from conv.related_net_duikang_beta import CNNEncoder_1234 as CNNEncoder
    from conv.related_net_duikang_beta import RelationNetwork_1234 as RelationNetwork
    from conv.related_net_duikang_beta import Domain_RelationNetwork_1234_0 as Domain_RelationNetwork
    from conv.related_op import conv_process_1234_test


feature_F_D,feature_F_S,feature_M_D,feature_M_S,F_S_COLLECTION,M_S_COLLECTION,F_D_COLLECTION,M_D_COLLECTION = load_kinface(args.dataset,args.dataroot)




def main():
    # step 1: init dataset
    
    save_file_name = "{}_dataset_loss_{}_conv_{}".format(args.addtion,args.dataset,str(K_PAIR),args.loss,args.conv)

    if args.is_save_log:
        from utils.log import get_logger 
        logger = get_logger(log_file=os.path.join(args.log_path,save_file_name+".log"))
    

#if args.dataset == 'KINFACE1':
    related_list = get_train_loader_kinface1(args.dataset,SELECT_SPLIT,F_S_COLLECTION,F_D_COLLECTION,M_D_COLLECTION,M_S_COLLECTION,BATCH_SIZE,args.picroot)
    train_loader = related_list[0]


    # init network
    print("init neural networks")
    torch.cuda.manual_seed(3407)
    feature_encoder_parents = CNNEncoder()
    feature_encoder_children = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)
    domain_relation_network = Domain_RelationNetwork(FEATURE_DIM, RELATION_DIM)
    MV = MVLoss(embedding_size=RELATION_DIM)


    if args.pretrained:
        feature_encoder_parents.load_state_dict(torch.load('KINFACE2_model/feature_encoder_parentsKINFACE1_KINFACE1_conv1234_MV_K_PAIR_1_SPLIT_5.pth'))
        feature_encoder_children.load_state_dict(torch.load('KINFACE2_model/feature_encoder_childrenKINFACE1_KINFACE1_conv1234_MV_K_PAIR_1_SPLIT_5.pth'))
        relation_network.load_state_dict(torch.load('KINFACE2_model/relation_network_KINFACE1_KINFACE1_conv1234_MV_K_PAIR_1_SPLIT_5.pth'))
        #domain_relation_network.load_state_dict(torch.load('KINFACE2_model/domainKINFACE1_KINFACE1_conv1234_MV_K_PAIR_1_SPLIT_1.pth'))


    feature_encoder_parents.cuda(GPU)
    feature_encoder_children.cuda(GPU)
    relation_network.cuda(GPU)
    domain_relation_network.cuda(GPU)
    MV.cuda(GPU)


    feature_encoder_parents_optim = torch.optim.Adam(feature_encoder_parents.parameters(), lr=LEARNING_RATE)
    feature_encoder_children_optim = torch.optim.Adam(feature_encoder_children.parameters(), lr=LEARNING_RATE)
    feature_encoder_parents_scheduler = StepLR(feature_encoder_parents_optim, step_size=1000, gamma=0.5)
    feature_encoder_children_scheduler = StepLR(feature_encoder_children_optim, step_size=1000,gamma=0.5)


    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=1000, gamma=0.5)
    domain_relation_network_optim = torch.optim.Adam(domain_relation_network.parameters(), lr=LEARNING_RATE)
    domain_relation_network_scheduler = StepLR(domain_relation_network_optim, step_size=1000, gamma=0.5)

    MV_loss_optim = torch.optim.Adam(MV.parameters(), lr=0.0005)
    MV_loss_scheduler = StepLR(MV_loss_optim, step_size=1000, gamma=0.5)


    print("training...")
    last_Mean = 0.0
    best_epoch = 0
    last_kinship_accuracy_for_FS = 0.0
    last_kinship_accuracy_for_FD = 0.0
    last_kinship_accuracy_for_MS = 0.0
    last_kinship_accuracy_for_MD = 0.0
    last_correct_fs = []
    last_incorrect_fs = []
    last_correct_fd = []
    last_incorrect_fd = []
    last_correct_ms = []
    last_incorrect_ms = []
    last_correct_md = []
    last_incorrect_md = []

    for episode in range(EPISODE):
        
        feature_encoder_parents.train()
        feature_encoder_children.train()
        relation_network.train()
        domain_relation_network.train()

        feature_encoder_parents_scheduler.step(episode)
        feature_encoder_children_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        domain_relation_network_scheduler.step(episode)
        MV_loss_scheduler.step(episode)
        # con
        # print(split)
        batch_features, batch_labels, batch_pairs = next(iter(train_loader))
        #print("domain_labels",batch_labels[1])
        combine=[]
        parent_set, child_set = batch_features.split(split_size=1, dim=1)
        for remove_num in range(parent_set.size(0)):#从0-batch_size依次
            res = list(range(parent_set.size(0)))#一个列表0-batch_size
            res.remove(remove_num)#移出去一个计数
            combine.append(random.sample(res,K_PAIR))#从剩余res中随机抽k_pair个，此时为1个，就是为了保证当前这个爹不要匹配上自己的儿子，匹配到其他任何一个儿子都可以

        child_list = []
        for i in combine:
            for j in i:
                child_list.append(j)
        # print("combine", combine)
        # print("child_list",child_list)#为了保证随机性
        # combine[[22], [24], [28], [14], [18], [30], [24], [9], [9], [14], [6], [30], [26], [11], [12], [22], [7], [5], [
        #     10], [6], [12], [29], [23], [6], [18], [26], [24], [29], [3], [21], [4], [21]]
        # child_list[
        #     22, 24, 28, 14, 18, 30, 24, 9, 9, 14, 6, 30, 26, 11, 12, 22, 7, 5, 10, 6, 12, 29, 23, 6, 18, 26, 24, 29, 3, 21, 4, 21]

        #实现每个父元素和多个子元素配对
        

        parent_features = Variable(parent_set.view(BATCH_SIZE, 3, 64, 64)).cuda(GPU).float()  # 32*1024
        child_features = Variable(child_set.view(BATCH_SIZE, 3, 64, 64)).cuda(GPU)  # k*312
        if args.conv == 'conv1234':
            parent_features1, parent_features2, parent_features3, parent_features4 = feature_encoder_parents(parent_features)
            child_features1, child_features2, child_features3, child_features4 = feature_encoder_children(child_features)

            p1,p2,p3,p4,c1,c2,c3,c4,new_domain_labels = conv_process_1234_test(parent_features1, parent_features2, parent_features3, parent_features4, child_features1, child_features2, child_features3, child_features4,child_list,batch_labels[1], K_PAIR, BATCH_SIZE, FEATURE_DIM,GPU)
            #print("new_domain_labels:",new_domain_labels)
            relation = relation_network(p1,p2,p3,p4,c1,c2,c3,c4)

            domain_relation = domain_relation_network(p1,p2,p3,p4,c1,c2,c3,c4)
            #print("shape of domain relation:",domain_relation.shape)

        labels_neg = torch.zeros(BATCH_SIZE * K_PAIR)  # 一堆全零的数值去和正例子拼
        labels = torch.cat((batch_labels[0], labels_neg), 0)  # 亲属关系的标签
        labels = labels.cuda(GPU)
        if args.loss == 'MV':
            labels = labels.long()

            distance = MV(relation,labels)
            # print("domain_relation",domain_relation)
            # print("domain_labels",domain_labels)
            from conv.loss_func import loss_final
            loss = loss_final(distance,labels,loss_type='HardMining',criteria=None) + loss_final(distance,labels,loss_type='FocalLoss',criteria=None)
        elif args.loss == 'CE':
            # labels = labels.unsqueeze(1)
            # print("shape of relation is",relation.shape)
            # print("shape of label is",labels.shape)
            relation = relation.float()
            labels = labels.squeeze().to(torch.int64)
            loss = F.cross_entropy(relation,labels)

        elif args.loss == 'triplet':
            relation_pos, relation_neg = relation[:BATCH_SIZE], relation[BATCH_SIZE:]
            # print("shape of relation_neg is", relation_neg.shape)
            # print("shape of relation_pos is", relation_pos.shape)
            #print(relation)#为什么全0啊
            loss = torch.sum(torch.clamp(torch.max(relation_neg,dim=1)[0]-torch.max(relation_pos,dim=1)[0]+0.5,0))/BATCH_SIZE

        if args.discriminator == True:
            new_batch_domain_labels = [4 if x == 10 else 5 if x == 11 else x for x in batch_labels[1]]
            new_batch_domain_labels_tensor = torch.tensor(new_batch_domain_labels)
            new_domain_labels_tensor = torch.tensor(new_domain_labels)

            # 使用torch.cat()拼接Tensor
            domain_labels = torch.cat((new_batch_domain_labels_tensor, new_domain_labels_tensor), 0)
            domain_labels = domain_labels.tolist()
            domain_labels = [0 if x == 1 else 1 if x == 2 else 2 if x == 4 else 3 if x == 5 else x for x in
                             domain_labels]
            domain_labels = torch.tensor(domain_labels)
            if args.discriminator_loss == 'ACE':
                unique_elements, counts = torch.unique(domain_labels, return_counts=True)
                # print("counts:",counts)
                num_classes = 4
                weights = torch.tensor([torch.sum(counts) / (num_classes * count) for count in counts]).cuda()
                # print("weights:",weights)
                domain_labels = domain_labels.long()
                domain_labels = domain_labels.cuda(GPU)
                loss = loss + F.cross_entropy(domain_relation, domain_labels,weights)
            elif args.discriminator_loss == 'CE':
                domain_labels = domain_labels.long()
                domain_labels = domain_labels.cuda(GPU)
                loss = loss + F.cross_entropy(domain_relation, domain_labels)


        # elif args.loss == 'triplet':
        #     relation_pos, relation_neg = relation[:32], relation[32:]
        #     anchor = random.choice(relation_pos)
        #     positive = random.choice(relation_pos)
        #     negative = random.choice(relation_neg)
        #     positive_distance = torch.abs(anchor - positive)
        #     negative_distance = torch.abs(anchor - negative)
        #     #print(positive_distance, negative_distance)
        #     loss = F.relu(positive_distance - negative_distance)

        # training

        feature_encoder_parents.zero_grad()
        feature_encoder_children.zero_grad()
        relation_network.zero_grad()
        MV_loss_optim.zero_grad()
        MV.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder_parents.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(feature_encoder_children.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

        feature_encoder_parents_optim.step()
        feature_encoder_children_optim.step()
        relation_network_optim.step()
        MV_loss_optim.step()


        if (episode) % args.eval_each_episode == 0:
            if not args.is_save_log:
                print("episode:", episode, "loss", loss.data)
                print("Parameters loss:{},conv:{},dataset:{},split:{:d},k_neg_pair:{:d},learning_rate:{:f}".format(args.loss,args.conv,args.dataset,SELECT_SPLIT,K_PAIR,LEARNING_RATE))
            

        if episode % args.eval_each_episode == 0:
    
            st = time.perf_counter()

            print("Testing...",str(episode))
            feature_encoder_parents.eval()
            feature_encoder_children.eval()
            relation_network.eval()
            domain_relation_network.eval()
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
                    test_loader = DataLoader(TEST_DATASET, batch_size=212, shuffle=True)
                    for features, test_labels, feature_pair in test_loader:
                        batch_size = test_labels.shape[0]

                        parent_set, child_set = features.split(split_size=1, dim=1)
                        # print(sample_SD.reshape(32,6272).shape)
                        # print(support_set.size())
                        parent_feature = Variable(parent_set.view(-1, 3, 64, 64)).cuda(GPU).float()  # 32*1024
                        child_feature = Variable(child_set.view(-1, 3, 64, 64)).cuda(GPU)  # k*312

                        if args.conv == 'conv1234':
                            parent_feature1, parent_feature2, parent_feature3, parent_feature4 = feature_encoder_parents(parent_feature)
                            child_feature1, child_feature2, child_feature3, child_feature4 = feature_encoder_children(child_feature)



                            # relations,x = relation_network(relation_pairs1,relation_pairs2,relation_pairs3,relation_pairs4)

                            relations = relation_network(parent_feature1,parent_feature2,parent_feature3,parent_feature4,child_feature1,child_feature2,child_feature3,child_feature4)
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
                if not args.is_save_log:
                    print('Mean:', Mean)
                    print('kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                        kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                        kinship_accuracy_for_MD))
                    print('best mean : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                        kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                        kinship_accuracy_for_MD))

                if Mean > last_Mean:
                    # save networks
                    last_Mean = Mean
                    last_kinship_accuracy_for_FS, last_correct_fs, last_incorrect_fs = kinship_accuracy_for_FS, correct_fs, incorrect_fs
                    last_kinship_accuracy_for_FD, last_correct_fd, last_incorrect_fd = kinship_accuracy_for_FD, correct_fd, incorrect_fd
                    last_kinship_accuracy_for_MS, last_correct_ms, last_incorrect_ms = kinship_accuracy_for_MS, correct_ms, incorrect_ms
                    last_kinship_accuracy_for_MD, last_correct_md, last_incorrect_md = kinship_accuracy_for_MD, correct_md, incorrect_md
                    if args.is_save_weights:
                            print("save networks for best-episode:", episode)
                            best_episode = episode
                            torch.save(feature_encoder_parents.state_dict(),join(args.modelpath,"feature_encoder_parents"+args.dataset+"_"+args.addtion +"_K_PAIR_"+str(K_PAIR)+"_SPLIT_"+ str(SELECT_SPLIT) + ".pth"))
                            torch.save(feature_encoder_children.state_dict(),join(args.modelpath,"feature_encoder_children" + args.dataset + "_" + args.addtion + "_K_PAIR_" + str(K_PAIR) + "_SPLIT_" + str(SELECT_SPLIT) + ".pth"))
                            torch.save(relation_network.state_dict(),join(args.modelpath, "relation_network_"+args.dataset+"_"+args.addtion +"_K_PAIR_"+str(K_PAIR)+"_SPLIT_"+ str(SELECT_SPLIT) +".pth"))
                            # torch.save(domain_relation_network.state_dict(),join(args.modelpath,"domain" + args.dataset + "_" + args.addtion + "_K_PAIR_" + str(K_PAIR) + "_SPLIT_" + str(SELECT_SPLIT) + ".pth"))
                            if args.loss == 'MV':
                                torch.save(MV.state_dict(),join(args.modelpath,"MVLoss" + args.dataset + "_" + args.addtion + "_K_PAIR_" + str(K_PAIR) + "_SPLIT_" + str(SELECT_SPLIT) + ".pth"))

                            print("保存了新权重")

            # if args.is_save_log:
            if args.is_save_log:
                logger.info("episode:{},loss,{}".format(str(episode),str(loss.data)))
                logger.info("Parameters loss:{},conv:{},dataset:{},split:{:d},k_neg_pair:{:d},learning_rate:{:f}".format(args.loss,args.conv,args.dataset,SELECT_SPLIT,K_PAIR,LEARNING_RATE))
                if args.dataset == 'txt':
                    print('Mean:', Mean)
                    print("best:",last_Mean)
                    logger.info('Mean :%.4f, best:%.4f' % (
                        Mean, last_Mean))
                else:
                
                    logger.info('kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                        kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                        kinship_accuracy_for_MD))
                    if  episode%(args.eval_each_episode*5)==0:
                        logger.info('best mean : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
                            kinship_accuracy_for_FS, kinship_accuracy_for_FD, kinship_accuracy_for_MS,
                            kinship_accuracy_for_MD))
        

    print('Mean:', last_Mean)
    if not os.path.exists(os.path.join(args.result_save_path,args.addtion+"_"+str(args.dataset)+"_"+str(K_PAIR)+".txt")):
        f = open(os.path.join(args.result_save_path,args.addtion+"_"+str(args.dataset)+"_"+str(K_PAIR)+"_a_"+".txt"),"w",encoding='utf-8')
        f.close()
    

    f = open(os.path.join(args.result_save_path,args.addtion+"_"+str(args.dataset)+"_"+str(K_PAIR)+"_a_"+".txt"),"a",encoding='utf-8')
    if args.dataset=='txt':
        dic = {'time': time.asctime( time.localtime(time.time())),"Mean":last_Mean}
    else:
        print('Mean:', last_Mean)
        print("best_episode:",best_episode)
        print('kinship : FS=%.4f, FD=%.4f, MS=%.4f, MD=%.4f' % (
            last_kinship_accuracy_for_FS, last_kinship_accuracy_for_FD, last_kinship_accuracy_for_MS,
            last_kinship_accuracy_for_MD))
        dic = {'time': time.asctime( time.localtime(time.time())),"Mean":last_Mean,"split":SELECT_SPLIT,"FS":last_kinship_accuracy_for_FS,'FD':last_kinship_accuracy_for_FD,'MS':last_kinship_accuracy_for_MS,"MD":last_kinship_accuracy_for_MD,'pkl':join(args.modelpath,args.dataset+"_"+args.addtion +"_K_PAIR_"+str(K_PAIR)+"_SPLIT_"+ str(SELECT_SPLIT) + ".pkl")}
        
    
    f.write(json.dumps(dic)+"\n")       
    f.close()

if __name__ == '__main__':
    main()

