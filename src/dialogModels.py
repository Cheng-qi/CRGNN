import torch
import torch.nn.functional as F
from torch import nn, no_grad
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, GCN2Conv, GCNConv, APPNP
import scipy.sparse as sp
import math
import numpy as np
from torch.nn.parameter import Parameter

class GATForDialog(torch.nn.Module):
    def __init__(self, input_size, head_num=3, hidden_size=64, out_size=4, dropout=0.5, **kwards) -> None:
        super(GATForDialog, self).__init__()
        self.head_num = head_num
        self.gconv1 = GATConv(input_size, hidden_size, self.head_num, dropout=dropout)
        self.act = nn.ReLU()
        self.gconv2 = GATConv(hidden_size*self.head_num, hidden_size, self.head_num, dropout=dropout)    
        self.gconv_out= GATConv(hidden_size*self.head_num, hidden_size, 1, concat=False, dropout=dropout)
        self.classfier = nn.Linear(hidden_size, out_size)
    def forward(self, x, edge_index, **kwards):
        log = {}
        out = self.gconv1(x, edge_index)
        out = self.act(out)
        out = self.gconv2(out, edge_index)
        out = self.act(out)
        out = self.gconv_out(out, edge_index)
        out = self.classfier(out)
        return out, log

class GCNForDialog(torch.nn.Module):
    def __init__(self, input_size, hidden_size, layers=64, out_size=4, dropout=0.6, residual = True, **kwards) -> None:
        super(GCNForDialog, self).__init__()
        self.gconvs = nn.ModuleList()
        self.gconvs.append(GCNConv(input_size, hidden_size))
        for i in range(layers-1):
            self.gconvs.append(GCNConv(hidden_size, hidden_size))
        self.classfier = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.act = nn.ReLU()
        
    def forward(self, x, edge_index, **kwards):
        h = x
        for i, con in enumerate(self.gconvs):
            h = self.dropout(h)
            h = con(h, edge_index)
            h = self.act(h)
        if self.residual:
            out = self.classfier(h+x)
        else:
            out = self.classfier(h)
        return out, {}

class GCNIIForDialog(torch.nn.Module):
    def __init__(self, input_size, hidden_size, layers=64, alpha=0.2, theta=0.5, out_size=4, dropout=0.6, **kwards) -> None:
        super(GCNIIForDialog, self).__init__()
        self.gconvs = nn.ModuleList()
        if input_size != hidden_size:
            self.project = nn.Linear(input_size, hidden_size)
        else:
            self.project = None
        for i in range(layers):
            self.gconvs.append(GCN2Conv(hidden_size, alpha, theta, i+1))
        self.classfier = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
    def forward(self, x, edge_index, **kwards):
        log = {}

        if self.project!=None:
            x = self.project(x)
        h_0 = x
        h = x
        for i, con in enumerate(self.gconvs):
            h = self.dropout(h)
            h = con(h, h_0, edge_index)
            h = self.act(h)
        out = self.dropout(h)
        out = self.classfier(out)
        return out,log

class MLPForDialog(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size, dropout=0.6, **kwards) -> None:
        super(MLPForDialog, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_size)
        )
    def forward(self, x, **kwards):
        log = {}
        out = self.MLP(x)
        return out, log


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

    def __init__(self, in_features, out_features, residual=False, 
                variant=False, 
                useAOR=True,
                useDLR=True):
        super(GraphConvolution, self).__init__() 
        self.useAOR = useAOR
        self.useDLR = useDLR
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

    def forward(self, input, adj , h0 , lamda, s, l):
        theta = lamda/l
        hi = torch.spmm(adj, input)
        # print(alpha)
        if not self.useAOR:
            support = hi
            r = support
        elif self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-s)*hi+s*h0
        else:
            # support = (1-alpha)*hi+h0
            support = (1-s)*hi+s*h0
            r = support
        if self.useDLR:
            output = theta*torch.mm(support, self.weight)+(1-theta)*r
        else: 
            output = torch.mm(support, self.weight)
        if self.residual:
            output = output+input
        return output



class ADGCNForDialog(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size,  
                layers, 
                dropout, 
                out_size,  
                useAOR=True,
                useDLR=True,
                variant=False,
                lamda=0.5,
                **kwards):
        super(ADGCNForDialog, self).__init__()
        if input_size != hidden_size:
            self.project = nn.Linear(input_size, hidden_size)
        else:
            self.project = None
        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GraphConvolution(hidden_size, hidden_size,variant=variant, useAOR=useAOR,
                useDLR=useDLR))
        self.layernorm = nn.LayerNorm(hidden_size)
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
            x = self.layernorm(self.project(x))
        layer_inner = x
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            if save_log: 
                log.update({"ADGCNLayer%d"%i:layer_inner})
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            s = torch.sigmoid(self.q(layer_inner)-1)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda, s, i+1))
            layer_inner = self.layernorm(layer_inner)
        if save_log: 
            log.update({"ADGCNLayer%d"%(i+1):layer_inner})
        
        # layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        out = self.classfier(layer_inner)
        return out, log