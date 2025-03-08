from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GINConv, GATConv
from src.gcn_conv import GCNConv
import random
import pdb

class global_pool_att(torch.nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, batch, keypoints):
        x = self.ln(x)
        out = []
        for i in range(batch.max()+1):
            x_i_weights = F.softmax((x[batch==i] * x[keypoints[i]]).sum(-1), -1)
            out.append((x[batch==i] * x_i_weights.view(-1, 1)).sum(0))
        return torch.stack(out, 0)

class CausalGAT(torch.nn.Module):
    def __init__(self, num_features,
                       num_classes, 
                       args, 
                       head=4, 
                       dropout=0.2,
                       **kwards):
        super(CausalGAT, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        if args.get("global_pool_type", "sum") == "att":
            self.global_pool = global_pool_att(hidden)
        elif args.get("global_pool_type", "sum") == "sum":
            self.global_pool = global_add_pool
        elif args.get("global_pool_type", "sum") == "mean":
            self.global_pool = global_mean_pool
        elif args.get("global_pool_type", "sum") == "max":
            self.global_pool = global_max_pool
        # self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=True, gfn=False)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.bn_feat = nn.LayerNorm(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = nn.LayerNorm(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = nn.LayerNorm(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        if args.use_residual:
            
            self.fc1_bn_o = nn.LayerNorm(hidden+num_features)
            self.fc1_o = Linear(hidden + num_features, hidden)
        else:
            self.fc1_bn_o = nn.LayerNorm(hidden)
            self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = nn.LayerNorm(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = nn.LayerNorm(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = nn.LayerNorm(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = nn.LayerNorm(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = nn.LayerNorm(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)


    def forward(self, x, edge_index, batch, keypoints, eval_random=True, **kward):

        # x = data.x if data.x is not None else data.feat
        # edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        if keypoints != None:
            keypoints_feats = x[keypoints]
        else:
            assert self.args.with_keypoints == False
            assert self.args.use_residual == False          
        x = self.bn_feat(x)

        
        #  gcn
        x = F.relu(self.conv_feat(x, edge_index)) # //gcn
        
        for i, conv in enumerate(self.convs): # laysers = 2
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))  # gcn
        
        
        # edge attention
        edge_rep = torch.cat([x[row], x[col]], dim=-1) # n_edge * (2 * node_dim)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1) # mlp shape: n_edge * 2， (10, 2) 
        edge_weight_c = edge_att[:, 0] # 蓝， 没用
        edge_weight_o = edge_att[:, 1] # 红， 有用 
        # edge_weight_c + edge_weight_o = [1, 1, 1, 1, 1, 1]
        
        attention = self.node_att_mlp(x)
        if self.args.with_keypoints:
            attention[keypoints, 0] = -1000000
            attention[keypoints, 1] = 1000000          
        node_att = F.softmax(attention, dim=-1)

        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # xc = self.global_pool(xc, batch)
        # xo = self.global_pool(xo, batch)
        if self.args.get("global_pool_type", "sum") == "att":
            xc = self.global_pool(xc, batch, keypoints) # 
            xo = self.global_pool(xo, batch, keypoints)
        else:
            xc = self.global_pool(xc, batch) # 
            xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        
        if self.args.use_residual:
            xo_logis = self.objects_readout_layer(torch.cat([xo, keypoints_feats], -1))
        else:
            xo_logis = self.objects_readout_layer(xo, keypoints_feats)
        # xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        return xc_logis, xo_logis, xo

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        # x_logis = F.log_softmax(x, dim=-1)
        x_logis = x
        return x_logis

    def objects_readout_layer(self, x, res=None):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)

        x = self.fc2_o(x)
        # x_logis = F.log_softmax(x, dim=-1)
        x_logis = x
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)
        
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        # x_logis = F.log_softmax(x, dim=-1)
        x_logis = x
        return x_logis
