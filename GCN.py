import numpy as np
import torch
import os
import torch.nn.functional as F
from torch.nn import Linear,Conv2d, MaxPool2d,ReLU
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_sort_pool, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,classification_report
from numpy import *

import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--ts', type=float, default=0.3)
parser.add_argument('--hc', type=int, default=15)
parser.add_argument('--trbs', type=int, default=80)
parser.add_argument('--tebs', type=int, default=7)
parser.add_argument('--lr', type=float, default=0.008)
args = parser.parse_args()

model_seed = 12345
seed = args.seed
test_size = args.ts
hidden_channels = args.hc
train_batch_size = args.trbs
test_batch_size = args.tebs
learing_rate = args.lr
epochs = 20

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
        #print("--------:",x_train.shape)
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


model = GCN(hidden_channels=hidden_channels)
print(model)


# 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCELoss()


# 网络训练及参数保存
def train():
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        loss_epoch = 0

        for data in train_loader:  # Iterate in batches over the training dataset.
            out, x_train = model(data.x, data.edge_index, data.edge_weight,
                                 data.batch)  # Perform a single forward pass. data.x 4500*15

            loss = criterion(out, data.y.float())  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            print(f'Epoch: {epoch:03d}, train_loss:{loss:.4f}')
    print('------------Finished Training------------')

    # 保存网络参数
    PATH = './GCN_final.pth'
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
        for data in test_loader:  # Iterate in batches over the training/test dataset.
            out, _ = model(data.x, data.edge_index, data.edge_weight, data.batch)
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

output = open('./GCN_final.pkl', 'wb')
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

with open("gcn_final.txt","a") as file:
    file.write("seed = %2.0f" % seed + '\n')
    file.write("model_seed = %2.0f" % model_seed + '\n')
    file.write("test_size = %2.1f" % test_size + '\n')
    file.write("hidden_channels = %2.0f" % hidden_channels + '\n')
    file.write("train_batch_size = %2.0f" % train_batch_size + '\n')
    file.write("test_batch_size = %2.0f" % test_batch_size + '\n')
    file.write("learing_rate = %2.6f" % learing_rate + '\n')
    file.write("epochs = %2.0f" % epochs + '\n')
    file.write('\n')

    file.write("TNR = %2.10f" % TNR + '\n')
    file.write("TPR = %2.10f" % TPR + '\n')
    file.write("AUC = %2.10f" % AUC + '\n')
    file.write("----------------------------------------------------" + '\n')
    file.flush()

