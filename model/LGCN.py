import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GCL import GraphConvolution


class LGCN(nn.Module):
    def __init__(self, adjs, dim_in, dim_out):
        super(LGCN,self).__init__()
        self.adjs = adjs
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.view_num = len(adjs)
        self.pai = nn.Parameter(torch.ones(self.view_num) / self.view_num, requires_grad=True)

        self.hidden_dim = dim_in // 2
    
        self.gc1 = GraphConvolution(self.dim_in, self.hidden_dim)
        self.gc2 = GraphConvolution(self.hidden_dim, self.dim_out)

        self.S = nn.Parameter(torch.randn_like(self.adjs[0]), requires_grad=True)
        self.theta = nn.Parameter(torch.FloatTensor([-5]).repeat(self.adjs[0].shape[0], 1), requires_grad=True) 


    def forward(self, X):
        exp_sum_pai = 0
        for i in range(self.view_num):
            exp_sum_pai += torch.exp(self.pai[i])

        weight = torch.zeros_like(self.pai)
        for i in range(self.view_num):
            weight[i] = torch.exp(self.pai[i]) / exp_sum_pai

        adj = weight[0] * self.adjs[0]
        for i in range(1, self.view_num):
            adj = adj + weight[i] * self.adjs[i]

        theta_sigmoid_tri = self.thred_proj(self.theta)

        S_add_ST = torch.sigmoid((self.S + self.S.t()) / 2)
        adj_S = adj * torch.relu(S_add_ST - theta_sigmoid_tri)  

        Z = torch.relu(self.gc1(X, adj_S))  
        Z = F.dropout(Z, p=0.3)
        Z = self.gc2(Z, adj_S)

        return Z
    
    def thred_proj(self, theta):
        theta_sigmoid = torch.sigmoid(theta)
        theta_sigmoid_mat = theta_sigmoid.repeat(1, theta_sigmoid.shape[0])
        theta_sigmoid_triu = torch.triu(theta_sigmoid_mat)
        theta_sigmoid_diag = torch.diag(theta_sigmoid_triu.diag())
        theta_sigmoid_tri = theta_sigmoid_triu + theta_sigmoid_triu.t() - theta_sigmoid_diag
        return theta_sigmoid_tri
