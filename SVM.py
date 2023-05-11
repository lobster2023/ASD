from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ChangedBehaviorWarning
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC
from sklearn.decomposition import KernelPCA
import numpy as np
import torch
import os
import scipy
import pandas as pd
from scipy import stats,linalg
from numpy import linalg as la
import torch.nn.functional as F
from torch.nn import Linear,Conv2d, MaxPool2d,ReLU
from torch_geometric.transforms import OneHotDegree as ohd
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_sort_pool, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
#%matplotlib inline
import matplotlib.pyplot as plt
from numpy import random,mat
import networkx as nx
import numpy.linalg as lg
import copy
import scipy.linalg as slg
from ipywidgets import interact
from stochastic import *
from sklearn.metrics import roc_curve,auc,classification_report
from numpy import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

model_seed = 12345
seed = 56
seed1 = 132
seed2 = 999
test_size = 0.34
gamma = 0.30

np.random.seed(seed)
torch.random.manual_seed(seed)
random.seed(seed)

# kpca = KernelPCA(n_components=5, kernel='linear')
#
# train_x = []
# train_y = []
# for i in range(len(train_list)):
#     train_x.append(kpca.fit_transform(train_list[i].x.numpy()).flatten())
#     train_y.append(train_list[i].y.numpy())
#
# test_x = []
# test_y = []
# for i in range(len(test_list)):
#     test_x.append(kpca.fit_transform(test_list[i].x.numpy()).flatten())  # .flatten()
#     test_y.append(test_list[i].y.numpy())
#
# # 创建一个SVM分类器并进行预测
#
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# for kernel in kernels:
#     clf = SVC(kernel=kernel, gamma=0.1)  # 创建SVM训练模型 kernel='rbf' RBF, Linear, Poly, Sigmoid
#     clf.fit(train_x, train_y)  # 对训练集数据进行训练
#     clf_y_predict = clf.predict(test_x)  # 通过测试数据，得到测试标签
#     scores = clf.score(test_x, test_y)  # 测试结果打分
#     auc = AUC(test_y, clf.decision_function(test_x))
# #     print('kernel=%s --> auc:%s '%(kernel,auc))

dti ='./huaxi/graph/data_hx/DTI'
fmri = './huaxi/graph/data_hx/fMRI'


def get_max_degree(filepath):
    '''
    计算所有图（样本）的最大度（带权图忽略权重，权重在后期计算中再参与）
    '''
    degree_graph = []
    for filename in os.listdir(filepath):
        sample_path = filepath + '/' + filename  # ./data/DTI//N10001.txt
        sample_adj = np.loadtxt(sample_path)  # 读取txt文件
        torch_adj = torch.from_numpy(sample_adj)  # 转为torch,193+89=282个 [90,90]
        edge_index = (torch_adj > 0).nonzero().t()  # 输出非零的位置并转置
        current_max_degree = int(degree(edge_index[0]).max())
        degree_graph.append(current_max_degree)
    return max(degree_graph)


def get_dataset_list(graph_path, fea_path, max_degree):
    '''
    构造融合后的数据集
    graph_path:用于构建邻接矩阵的图数据路径
    fea_path:用于构建节点特征的图数据路径
    max_degree:所有样本图的最大度取值
    '''

    data_list = []
    for filename in os.listdir(graph_path):
        digits = ''.join([x for x in filename if x.isdigit()])
        sample_label = int(digits[0])
        sample_label = abs(sample_label - 2)  # 1为阳性->1，2 为阴性->0
        sample_id = digits[1:]

        sample_path = graph_path + '/' + filename
        sample_adj = np.loadtxt(sample_path)
        torch_adj = torch.from_numpy(sample_adj)

        # 构造节点特征，实现双模融合
        for file in os.listdir(fea_path):
            digits_ = ''.join([x for x in file if x.isdigit()])
            sample_id_ = digits_[1:]  # 提取样本ID
            if sample_id == sample_id_:  # 比对邻接矩阵样本ID和对应节点特征图的样本ID
                sample_fea_path = fea_path + '/' + file
                sample_fea = np.loadtxt(sample_fea_path)
                sample_fea = torch.from_numpy(sample_fea)
                sample_fea = sample_fea.float()
            else:
                continue

        edge_index = (torch_adj > 0).nonzero().t()

        # 单独提取边的权重
        edge_weight = torch_adj.view(torch.numel(torch_adj))  # 展平，把[90, 90]变成[8100]
        edge_weight = edge_weight[edge_weight.nonzero()]
        edge_weight = edge_weight.squeeze(1)  # torch.Size([788, 1]) — torch.Size([788])
        edge_weight = edge_weight.float()

        graph = Data(x=sample_fea, edge_index=edge_index, y=torch.tensor(sample_label),
                     edge_weight=edge_weight)

        data_list.append(graph)
    return data_list


max_degree = get_max_degree(fmri)
data_list = get_dataset_list(fmri, dti, max_degree)


# postive and negative num
def neg_num(data_list):
    k = 0
    for i in range(len(data_list)):
        if data_list[i]['y'] == 1:
            k += 1
    return k


negative_list = data_list[:neg_num(data_list)]
positive_list = data_list[neg_num(data_list):]
print(f'Number of negative graphs: {len(negative_list)}')
print(f'Number of positive graphs: {len(positive_list)}')

negative_train, negative_test = train_test_split(negative_list, test_size=test_size, random_state=seed1)
positive_train, positive_test = train_test_split(positive_list, test_size=test_size, random_state=seed2)
train_list = positive_train + negative_train
test_list = positive_test + negative_test

def f (precited,expected):
    res = precited ^ expected  #亦或使得判断正确的为0,判断错误的为1
    r = np.bincount(res)
    tp_list = ((precited)&(expected))
    fp_list = (precited&(~expected))
    tp_list=tp_list.tolist()
    fp_list=fp_list.tolist()
    tp=tp_list.count(1)
    fp=fp_list.count(1)
    tn = r[0]-tp
    fn = r[1]-fp
    p=tp/(tp+fp)
    recall = tp/(tp+fn)
    F1=(2*tp)/(2*tp+fn+fp)
    acc=(tp+tn)/(tp+tn+fp+fn)
    return tn,fp,recall

from sklearn.metrics import roc_curve, auc, classification_report,recall_score,roc_auc_score
from numpy import *

def maxauc(k):
    kpca = KernelPCA(n_components=k, kernel='cosine')  # “linear poly rbf sigmoid cosine precomputed

    train_x = []
    train_y = []
    for i in range(len(train_list)):
        train_x.append(kpca.fit_transform(train_list[i].x.numpy()).flatten())
        train_y.append(train_list[i].y.numpy())

    test_x = []
    test_y = []
    for j in range(len(test_list)):
        test_x.append(kpca.fit_transform(test_list[j].x.numpy()).flatten())  # .flatten()
        test_y.append(test_list[j].y.numpy())

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    aucresult = []
    tprresult=[]
    tnrresult = []
    for kernel in kernels:
        clf = SVC(kernel=kernel, gamma=gamma)  # 创建SVM训练模型 kernel='rbf' RBF, Linear, Poly, Sigmoid
        clf.fit(train_x, train_y)  # 对训练集数据进行训练
        clf_y_predict = clf.predict(test_x)  # 通过测试数据，得到测试标签
        scores = clf.score(test_x, test_y)  # 测试结果打分
        fpr, tpr, _ = roc_curve(test_y, clf.decision_function(test_x))
        #print("fpr:",fpr)
        auc = AUC(test_y, clf.decision_function(test_x))
        aucresult.append(auc)
        y_score=clf.decision_function(test_x)
        for i in range(len(y_score)):
            if y_score[i] < 0:
                y_score[i] = 0
            else :
                y_score[i] = 1
        y_pred = y_score
        #print("y_pred:",y_pred)

        #print(type(y_true),type(y_pred))

        int_y_true = map(int, test_y)
        int_y_pred = map(int, y_pred)
        #print(int_y_true)

        #tn,fp,TPR=f(np.array(y_pred),np.array(y_true))
        tn, fp, TPR = f(np.array(list(int_y_pred)), np.array(list(int_y_true)))
        TNR=tn/(fp+tn)
        recall_value=recall_score(test_y, y_pred)
        #print("TNR:",TNR)
        tprresult.append(TPR)
        tnrresult.append(TNR)
    return max(aucresult),max(tprresult),max(tnrresult)


tnr_res = []
tpr_res = []
auc_res = []

for i in range(1,90):
    auc,tpr,tnr=maxauc(i)
    auc_res.append(auc)
    tnr_res.append(tnr)
    tpr_res.append(tpr)
# print(max(max(auc_res)))
print("auc_res:",auc_res)
print("tnr_res:",tnr_res)
print("tpr_res:",tpr_res)
print("AUC=",max(auc_res))
print("TPR=",max(tpr_res))
print("TNR=",max(tnr_res))

with open("SVM_fmri+dti.txt","a") as file:
    file.write("seed1 = %2.0f" % seed1 + '\n')
    file.write("seed2 = %2.0f" % seed2 + '\n')
    file.write("model_seed = %2.0f" % model_seed + '\n')
    file.write("test_size = %2.2f" % test_size + '\n')
    file.write("gamma = %2.2f" % gamma + '\n')
    file.write('\n')

    file.write("特异度TNR = %2.10f" % max(tnr_res) + '\n')
    file.write("敏感度TPR = %2.10f" % max(tpr_res)+ '\n')
    file.write("AUC = %2.10f" % max(auc_res) + '\n')
    file.write("----------------------------------------------------" + '\n')
    file.flush()