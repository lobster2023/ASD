import torch.nn as nn
#from torch_geometric.utils import scatter
import torch_geometric.nn as pyg_nn

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

import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--ts', type=float, default=0.2)
parser.add_argument('--hc', type=int, default=16)
parser.add_argument('--trbs', type=int, default=69)
parser.add_argument('--tebs', type=int, default=9)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--a', type=float, default=0.1)
args = parser.parse_args()

model_seed = 12345
seed = args.seed
test_size = args.ts
hidden_channels = args.hc
train_batch_size = args.trbs
test_batch_size = args.tebs
learing_rate = args.lr
epochs = 20
a_lambda = args.a


np.random.seed(seed)
torch.random.manual_seed(seed)
random.seed(seed)

#smri = './huaxi/graph/data_hx/sMRI'
dti ='./huaxi/graph/data_hx/DTI'
fmri = './huaxi/graph/data_hx/fMRI'


def get_max_degree(filepath):
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

max_degree = get_max_degree(dti)
data_list = get_dataset_list(dti, fmri, max_degree) #DTI+fMRI:效果较好 74%
# data_list = get_dataset_list(dti, smri, max_degree) #DTI + sMRI:效果较差 71%

#resuffle the list
data_list1,data_list2 = train_test_split(data_list, test_size=test_size, random_state=seed)
data_list=data_list1+data_list2

data_y=[]
for i in range(len(data_list)):
    data_y.append(float(data_list[i].y.detach().numpy()))
print("data_y:",data_y)

from torch_geometric.nn import GATConv, ChebConv, GCNConv, SAGEConv
class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.k = 70
        torch.manual_seed(seed)
        num_heads = 4  # 每个节点的特征数

        self.conv1 = GATConv(90,hidden_channels //num_heads, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels,hidden_channels //num_heads, heads=num_heads, concat=True)

        self.lin1 = Linear(hidden_channels * self.k, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, data, batch):
        x, edge_index,edge_weight = data.x, data.edge_index,data.edge_weight
        x = self.conv1(x, edge_index)
        x_train = x
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_sort_pool(x, batch, self.k)  # [batch_size, hidden_channels*self.k]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        x = torch.squeeze(x)

        return x, x_train


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
device = 'cpu'  # 设备
num_node_features = 90  # 每个节点的特征数
num_classes = 2 # 每个节点的类别数
data = data_list[0].to(device)  # Cora的一张图
# 4.定义模型
model = GATNet().to(device)
print(model)

#optimizer = torch.optim.SGD(model.parameters(),momentum=0.9,nesterov=True, lr=learing_rate, weight_decay=1e-3)  # 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate, weight_decay=1e-3)  # 优化器
#loss_function = nn.NLLLoss()  # 损失函数
criterion = torch.nn.BCELoss()

def wass_dist_(A, B):
    n = len(A)
    l1_tilde = A + np.ones([n,n])/n #adding 1 to zero eigenvalue; does not change results, but is faster and more stable
    l2_tilde = B + np.ones([n,n])/n
    s1_tilde = lg.inv(l1_tilde)
    s2_tilde = lg.inv(l2_tilde)
    Root_1= slg.sqrtm(s1_tilde)
    Root_2= slg.sqrtm(s2_tilde)
    return np.trace(s1_tilde) + np.trace(s2_tilde) - 2*np.trace(slg.sqrtm(Root_1 @ s2_tilde @ Root_1))

# x.detach().numpy() 将x转换成tensor型
def dis2(x, label, N1, N2):
    s = 0
    batch_sample = np.vsplit(x.detach().cpu().numpy(), int(x.detach().cpu().numpy().shape[0]) / 90)  # 将样本横向分割成batch*90*15 len(batch_sample)=batch array型

    sample = []
    S1 = 0
    S2 = 0
    S = 0
    for i in range(len(batch_sample)):
        ls = batch_sample[i]  # batch_sample[i]为array 90*15
        ls_t = mat(ls).T
        result = ls_t * mat(ls)  # 90*90
        sample.append(result)  # 正负样本 batch个

    for i in range(len(sample)):
        for j in range(1, i + 1):
            if label[i] == 1 and label[j] == 1:
                w1 = np.power(wass_dist_(sample[i], sample[j]), 2) / N1
                S1 = S1 + w1
            if label[i] == 0 and label[j] == 0:
                w2 = np.power(wass_dist_(sample[i], sample[j]), 2) / N2
                S2 = S2 + w2
            else:
                w = np.power(wass_dist_(sample[i], sample[j]), 2) / (N1 * N2)
                S = S + w

    s = (S1 + S2) / S
    return s

def train():
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            out, x_train= model(data,data.batch.to(device))  # Perform a single forward pass. data.x 4500*15
            N1 = int((1 == data.y).sum())
            N2 = int((0 == data.y).sum())
            loss1 = dis2(x_train, data.y.to(device), N1, N2)  # data.y：正负样本标签

            loss2 = criterion(out, data.y.float().to(device))  # Compute the loss.
            # print(data.y)
            # loss_epoch=loss_epoch+float(loss)
            a = a_lambda
            loss = a * loss1 + (1 - a) * loss2
            # loss_epoch=loss_epoch+float(loss)

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            print(f'Epoch: {epoch:03d}, train_loss: {loss:.4f}')
    print('------------Finished Training------------')

    # 保存网络参数
    PATH = './GAT+loss_final.pth'
    torch.save(model.state_dict(), PATH)
    return model, PATH


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

# 10折交叉验证
OUT_TOTAL = []
n=0
for i in range(0, 10):
    test_list = data_list[n*14:(n+1)*14]
    train_list = [val for val in data_list if val not in test_list]
    n=n+1
    train_loader = DataLoader(train_list, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=test_batch_size, shuffle=False)
    test_total_acc = []
    model, PATH = train()

    model.load_state_dict(torch.load(PATH))
    from sklearn.metrics import roc_curve, auc, classification_report,roc_auc_score,recall_score
    from numpy import *


    with torch.no_grad():
        model.eval()
        correct = 0
        print("----------------------------")
        # num_heads = 4
        for data in test_loader:  # Iterate in batches over the training/test dataset.
            out, _ = model(data,data.batch.to(device))
            #print("-------------", out, "---------------")
            OUT_TOTAL = concatenate((OUT_TOTAL, out.numpy()), axis=0)
            #OUT_TOTAL.append(float(out))
            #print("OUT_TOTAL:", OUT_TOTAL)
            print("-------------", "第",len(OUT_TOTAL),"个样本训练完毕", "---------------")



y_true=data_y
y_score=OUT_TOTAL

AUC=roc_auc_score(y_true, y_score)
print("AUC:",AUC)

fpr, tpr, thresh_hold = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
#print(fpr,tpr)

roc_auc = auc(fpr, tpr)
output = open('./gat+loss_final.pkl', 'wb')
pickle.dump([fpr, tpr, roc_auc], output, -1)
output.close()

for i in range(len(y_score)):
    if y_score[i] < 0.5:
        y_score[i] = 0
    else:
        y_score[i] = 1
y_pred = y_score
#print("y_pred:", y_pred)

int_y_true = map(int, y_true)
int_y_pred = map(int, y_pred)

#tn,fp,TPR=f(np.array(y_pred),np.array(y_true))
tn, fp, TPR = f(np.array(list(int_y_pred)), np.array(list(int_y_true)))
TNR=tn/(fp+tn)


print("groudtruth:", y_true)
print("predicted labels:", y_score)
print("特异度TNR = %2.10f" % TNR)
print("敏感度TPR = %2.10f" % TPR)  # 最关注！！！ 80%
print("AUC = %2.10f" % AUC)

with open("GAT+loss_final.txt","a") as file:
    file.write("seed = %2.0f" % seed + '\n')
    file.write("model_seed = %2.0f" % model_seed + '\n')
    file.write("test_size = %2.1f" % test_size + '\n')
    file.write("hidden_channels = %2.0f" % hidden_channels + '\n')
    file.write("train_batch_size = %2.0f" % train_batch_size + '\n')
    file.write("test_batch_size = %2.0f" % test_batch_size + '\n')
    file.write("learing_rate = %2.6f" % learing_rate + '\n')
    file.write("epochs = %2.0f" % epochs + '\n')
    file.write("a_lambda = %2.2f" % a_lambda + '\n')
    file.write('\n')

    file.write("TNR = %2.10f" % TNR + '\n')
    file.write("TPR = %2.10f" % TPR + '\n')
    file.write("AUC = %2.10f" % AUC + '\n')
    file.write("----------------------------------------------------" + '\n')
    file.flush()

