
import scipy.sparse as sp
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import seaborn as sns
import matplotlib.pyplot as plt

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sys_normalized_adjacency(adj):
   device = adj.device
   adj = adj.data.cpu().numpy()
   data = [1.]* adj.shape[1]
   row = adj[0]
   col = adj[1]
   adj = sp.coo_matrix((data, (row, col)), shape=(max(adj[0])+1, max(adj[0])+1))
#    adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return sparse_mx_to_torch_sparse_tensor(d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).to(device)
   
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        # print(alpha)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            # support = (1-alpha)*hi+h0
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output



class ADGCNForDialog(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size,  
                layers, 
                dropout, 
                lamda, 
                out_size,  
                variant=False,
                **kwards):
        super(ADGCNForDialog, self).__init__()
        if input_size != hidden_size:
            self.project = nn.Linear(input_size, hidden_size)
        else:
            self.project = None
        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GraphConvolution(hidden_size, hidden_size,variant=variant))
        self.act_fn = nn.ReLU()
        self.q = nn.Linear(hidden_size, 1, bias = True)
        self.dropout = dropout
        # self.alpha = alpha
        self.lamda = lamda
        self.classfier = nn.Linear(hidden_size, out_size)

    def forward(self, x, adj, save_log=False, **kwards):
        log = {}
        adj = sys_normalized_adjacency(adj)
        _layers = []
        # x = F.dropout(x, self.dropout, training=self.training)
        if self.project != None:
            x = self.project(x)
        layer_inner = x
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            if save_log: 
                log.update({"ADGCNLayer%d"%i:layer_inner})
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            alpha = torch.sigmoid(self.q(layer_inner)-1)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda, alpha, i+1))
        if save_log: 
            log.update({"ADGCNLayer%d"%(i+1):layer_inner})
        layer_inner = F.dropout(layer_inner+x, self.dropout, training=self.training)
        out = self.classfier(layer_inner)
        return out, log