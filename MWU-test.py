import seaborn as sns
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

model_seed = 12345
seed = 1
test_size = 0.2
hidden_channels = 15
train_batch_size = 68
test_batch_size = 7
learing_rate = 0.008
epochs = 20
a_lambda = 0.1

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



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(model_seed)
        self.k = 70
        self.conv1 = GCNConv(90, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels * self.k, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x_train = x
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        # print(x.shape)    #5760*20

        # 2. Readout layer
        x = global_sort_pool(x, batch, self.k)  # [batch_size, hidden_channels*self.k]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        x = torch.squeeze(x)

        return x, x_train

device = 'cpu'

model = GCN(hidden_channels=hidden_channels).to(device)
print(model)


# 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCELoss()

# def wass_dist_(A, B):
#     n = len(A)
#     l1_tilde = A + np.ones([n,n])/n #adding 1 to zero eigenvalue; does not change results, but is faster and more stable
#     l2_tilde = B + np.ones([n,n])/n
#     s1_tilde = lg.inv(l1_tilde)
#     s2_tilde = lg.inv(l2_tilde)
#     Root_1= slg.sqrtm(s1_tilde)
#     Root_2= slg.sqrtm(s2_tilde)
#     return np.trace(s1_tilde) + np.trace(s2_tilde) - 2*np.trace(slg.sqrtm(Root_1 @ s2_tilde @ Root_1))

# x.detach().numpy() 将x转换成tensor型
def dis2(x, label, N1, N2):

    batch_sample = np.vsplit(x.detach().cpu().numpy(), int(x.detach().cpu().numpy().shape[0]) / 90)  # 将样本横向分割成batch*90*15 len(batch_sample)=batch array型

    sample = []

    for i in range(len(batch_sample)):
        ls = batch_sample[i]  # batch_sample[i]为array 90*15
        ls_t = mat(ls).T
        result = mat(ls) * ls_t  # 90*90
        print("**********result*********:",result.shape)
        sample.append(result)  # 正负样本 batch个

    return sample



# 网络训练及参数保存
def train():
    model.train()
    loss_epoch = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        out, x_train = model(data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),
                             data.batch.to(device))  # Perform a single forward pass. data.x 4500*15
        N1 = int((1 == data.y).sum())
        N2 = int((0 == data.y).sum())
        sample = dis2(x_train, data.y.to(device), N1, N2)  # data.y：正负样本标签
        print("------len of sample-----", len(sample))
        print("------sample-----", sample)

        loss = criterion(out, data.y.float().to(device))  # Compute the loss.
        print(data.y)
        # loss_epoch=loss_epoch+float(loss)

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    print('------------Finished Training------------')

    # 保存网络参数
    PATH = './ks_test.pth'
    torch.save(model.state_dict(), PATH)
    return model, PATH,sample


# 10折交叉验证
OUT_TOTAL = []
n=0
a=0
for i in range(0, 10):
    test_list = data_list[n*14:(n+1)*14]
    train_list = [val for val in data_list if val not in test_list]
    n=n+1
    train_loader = DataLoader(train_list, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=test_batch_size, shuffle=False)
    test_total_acc = []
    model, PATH,sample = train()
    print("------len of sample-----", len(sample))
    print("------sample-----", sample)
    count1 = 0
    count0 = 0
    # 放最初的特征值
    d1 = []
    d0 = []

    #sample_path = filepath + '/' + filename
    # 广义邻接矩阵
    sample_adj = np.array(sample)
    print("sample_adj:",sample_adj[0].shape,sample_adj[0])
    print("*****************len of sample_adj:*****************", len(sample_adj))
    # 标签
    # digits = ''.join([x for x in filename if x.isdigit()])
    # sample_label = int(digits[0])
    # sample_label = abs(sample_label - 2)  # 1为阳性->1，2 为阴性->0
    # sample_id = digits[1:]

    print("data_y:",data_y)
    test_label = data_y[a * 14:(a + 1) * 14]
    print("test_label:",test_label)
    print("##########",data_y[0:a * 14])
    print("##########", data_y[(a+1) * 14:137])

    train_label = concatenate((data_y[0:a * 14],data_y[(a+1) * 14:137]), axis=0)
    a = a + 1
    sample_label = train_label
    print("train_label",train_label)
    print("**************************************************---------------------------------------------------------------------------------------", '\n', sample_label)
    a = 0
    for i in range(len(sample_adj)):

        # 广义度矩阵
        sample_deg = np.diag(np.sum(sample_adj[i], axis=1))
        # 拉普拉斯矩阵
        print("sample_adj.type:",type(sample_adj))
        print("sample_adj.shape:",sample_adj[0].shape)
        print("sample_deg.shape:",sample_deg.shape)

        sample_lap = sample_deg - sample_adj[i]
        # 特征值
        sample_eigvalue = np.linalg.eig(sample_lap)[0]

        print("---------------------------------------------------------------------------------------",'\n',sample_label)

        print("---------------------------------------------------------------------------------------",'\n',type(sample_label),sample_label)
        print("i=",i)

    #print(sample_label[i])
    # 把特征值和标签融合
        sam_label=sample_label[a]
        if sam_label == 1:
            count1 += 1
            d1.append(sample_eigvalue)
        else:
            count0 += 1
            d0.append(sample_eigvalue)
        # 将矩阵转换为第一个矩阵为相应标签的第一个特征值
        a=a+1
    d0x = []
    d1x = []
    for i in range(90):
        d0x.append([])
        for j in range(count0):
            d0x[i].append(d0[j][i])
    for i in range(90):
        d1x.append([])
        for j in range(count1):
            d1x[i].append(d1[j][i])





print("Dti result")
print("***",len(d1x),d1x)
print("###",len(d0x),d0x)



#eig_value(dti)[0][0:2]
datap=[d0x[1], d1x[1]]

plt.figure(figsize=(15,8))
plt.style.use('default')
# plt.style.use('seaborn-colorblind')
sns.kdeplot(datap[0],color='coral',linewidth = 2.5)
sns.kdeplot(datap[1],color='steelblue',linewidth = 1.5)
plt.legend(['Positive','Negative'])
plt.title('Eigenvalue Kernel Density Graph of Samples')
plt.savefig('KS-test_00.')

#DTI+fMRI
# datap=[eig_value(dti)[0][1], eig_value(dti)[1][1]]

plt.figure(figsize=(180,120))
for i in range(90):
    plt.subplot(9,10, i+1)
    p1=sns.kdeplot(d0x[i], color="coral",linewidth = 6.5)
    p1=sns.kdeplot(d1x[i], color="steelblue",linewidth = 5.5)
    plt.legend(['Positive','Negative'])
    plt.rc('legend',fontsize=20)
    plt.title(i+1,y=-0.1,fontsize=60)

plt.savefig('KS-test_all.',dpi=300)

# plt.show()

d0x = np.array(d0x)
d1x = np.array(d1x)

# 均值假设检验
p_value = []
d_value = []
result = []

for i in range(90):
    #         p_value.append(np.average(stats.ks_2samp(d0x[i],d1x[i])[1]))
    #         d_value.append(np.average(stats.ks_2samp(d0x[i],d1x[i])[0]))
    p_value.append(np.average(stats.mannwhitneyu(d0x[i], d1x[i])[1]))
    d_value.append(np.average(stats.mannwhitneyu(d0x[i], d1x[i])[0]))

# 将id p t加入结果中
for i, v in enumerate(p_value):
    result.append((i, v, d_value[i]))
result.sort(key=lambda x: x[1], reverse=True)
result1 = pd.DataFrame(result, columns=["node", "p-value", "d-value"])

result1.to_csv("KS-test_00.csv")
