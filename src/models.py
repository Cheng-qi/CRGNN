import torch
import torch.nn.functional as F
from torch import nn, no_grad
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
# from src.ADGCN import ADGCNForDialog
import torchvision
import src.CausalGAT as causalGNNs
import src.dialogModels as dialogModels
from src.dialogModels import *
import numpy as np
from src.emotionFroms3pl import *
from src.tfcnn import *
from src.dst.dst import DST
from src.DWFormer.model import DWFormer
# from src.LabelAdaptiveMixup.model import SpeechModel as labelAdaptiveMixup
from src.LabelAdaptiveMixup.center_loss import CenterLoss
class MyModel(nn.Module):
    def loss(self, classes, labels, mode="classification"):
        # return F.cross_entropy(classes, labels, weight=torch.tensor([1.0,2.0,2.0,3.0]).to(classes.device))
        if mode == "classification":
            if self.cfg.get("used_focal_loss", False):
                return torchvision.ops.sigmoid_focal_loss(classes, F.one_hot(labels, len(self.cfg.classes_name)).float().to(labels.device), reduction="mean")
            return F.cross_entropy(classes, labels, torch.FloatTensor(self.cfg.get("loss_weight", None)).to(labels.device) if self.cfg.get("loss_weight", None)!=None else None)
        else:
            return F.mse_loss(classes.view(-1), labels)        
    def metric(self, out, labels):
        out = out.data.cpu().numpy()
        pred_out = out.argmax(-1)
        labels = labels.cpu().numpy()
        classes = self.cfg.classes_name
        accs = []
        nums = []

        single_accuracy = ""
        for i, class_name in enumerate(classes):
            index_i = np.argwhere(labels == i)
            if len(index_i)==0:
                accs.append(0)
                nums.append(0)
                continue
            else:
                accs.append(accuracy_score(labels[index_i], pred_out[index_i]))
                nums.append(len(index_i))
            single_accuracy = single_accuracy + "%s:%.4f,"%(class_name, accs[-1])
        unweight_accuracy = sum(accs)/len(classes)

        accuracy = accuracy_score(labels, pred_out)
        f1_micro = f1_score(labels, pred_out, average="micro")
        f1_macro = f1_score(labels, pred_out, average="macro")
        f1_weighted = f1_score(labels, pred_out, average="weighted")
        # c_m = confusion_matrix(labels, pred_out)
        res = {
            "unweight_accuracy":unweight_accuracy, 
            "single_accuracy":single_accuracy,
            "accuracy":accuracy, 
            "f1_micro":f1_micro, 
            "f1_macro":f1_macro,
            "f1_weighted":f1_weighted
            # "c_m":torch.Tensor(c_m),
        }
        return res
def build_intra_graph(dialog_lengths, to_future_link=4, to_past_link=0, device="cpu"):
    # build intra graph to update node embeding from same model
    batch_size = dialog_lengths.shape[0]
    graphs = []
    feats = []
    for i in range(batch_size):
        dialog_length = dialog_lengths[i]
        adj = torch.eye(dialog_length)
        for ii in range(dialog_length):
            adj[ii,ii+1:(min(ii+to_future_link+1, dialog_length))] = 1
            adj[ii,(max(ii-to_past_link, 0)):ii] = 1
        graphs.append(adj)
    all_adj = torch.zeros(dialog_lengths.sum(), dialog_lengths.sum())
    for i in range(batch_size):
        start = dialog_lengths[:i].sum()
        end = start + dialog_lengths[i]
        all_adj[start:end, start:end] = graphs[i]
    all_adj = torch.stack(torch.where(all_adj!=0),dim = 0).to(device)
    return all_adj
def build_context_graph(
    dialog_lengths, 
    context_before=4, 
    context_after=4, 
    device="cpu"):
    
    node_indexes = []
    batches = []
    key_points = []
    edges_list = torch.empty(2,0)
    acc_uttr = 0
    for dialog_i in dialog_lengths:
        
        indexes_group = [list(range(max(i-context_before,0), min(i+context_after+1, dialog_i))) for i in range(dialog_i)]
        # batch_start = max(batches)+1 if len(batches)==0 else 0
        for con_i, context in enumerate(indexes_group):
            
            key_points.append(min(con_i, context_before)+len(batches))
            node_indexes.extend([acc_uttr+i for i in context])
            # node_indexes.extend([(max(batches)+1 if len(batches)!=0 else 0)+i for i in context])
            edges_list =  torch.concat([edges_list, torch.stack(torch.where(torch.ones(len(context), len(context))!=0)) + len(batches)],1)
            batches.extend(len(context)*[max(batches)+1 if len(batches)!=0 else 0])
        acc_uttr += int(dialog_i)
    return torch.LongTensor(node_indexes).to(device), edges_list.to(torch.long).to(device),  torch.LongTensor(batches).to(device) ,  torch.LongTensor(key_points).to(device)
        # [batches.extend(len(context)*[batch_start+i]) for i, context in enumerate(indexes_group)]
    
class DiaModel(MyModel):
    def __init__(self, cfg, **kwards):
        super().__init__()
        self.cfg = cfg
        graph_input_size = cfg.uttr_input_dim
        if cfg.get("adapter", None) != None:
            self.adapter = AdapterModel(input_dim = cfg.uttr_input_dim, **cfg.adapter)
            graph_input_size = self.cfg.adapter.output_dim
        self.dialogModel = \
            getattr(dialogModels, 
                    cfg.dialogModel.name)\
                        (input_size = graph_input_size,
                        out_size = len(cfg.classes_name),
                        **cfg.dialogModel)
    def forward(self,  frames_inputs=None, frames_lengths=None, uttr_input=None, dialog_lengths=None, **kwards):
        res = {}
        # extract features
        if self.cfg.get("adapter", None) != None:
            uttr_input, _= self.adapter(frames_inputs, frames_lengths)
        graph_adj = build_intra_graph(dialog_lengths, 
            to_future_link=self.cfg.dialogModel.to_future_link, 
            to_past_link=self.cfg.dialogModel.to_past_link, 
            device = dialog_lengths.device)
        x, log = self.dialogModel(uttr_input, graph_adj)
        res.update(log)
        return x, res

class CausalIntraDiaModel(MyModel):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.causalGNN.args.with_keypoints = False
        self.cfg.causalGNN.args.use_residual = False
        causalGNN_input_dim = cfg.uttr_input_dim
        if cfg.get("adapter", None) != None:
            self.adapter = AdapterModel(input_dim = cfg.uttr_input_dim, **cfg.adapter)
            causalGNN_input_dim = self.cfg.adapter.output_dim
        
        self.causalGNN = \
            getattr(causalGNNs, 
                    cfg.causalGNN.name)\
                        (num_features = cfg.uttr_input_dim,
                         num_classes = len(cfg.classes_name), 
                         **cfg.causalGNN)
                        
        if self.cfg.get("residual", True):
            self.residual = nn.Sequential(
                nn.Linear(causalGNN_input_dim, cfg.causalGNN.args.hidden),
                nn.ReLU()
            )
        # dialog model
        self.dialogModel = \
            getattr(dialogModels, 
                    cfg.dialogModel.name)\
                        (
                            input_size = cfg.causalGNN.args.hidden,
                            out_size = len(cfg.classes_name),
                            **cfg.dialogModel)
        self.cache_graph = {}
    def forward(self, frames_inputs=None, frames_lengths=None, uttr_input=None, dialog_lengths=None, is_eval=False, epoch=0,**kwards):
        log = {}
        if self.cfg.get("adapter", None) != None and self.cfg.get("residual", True):
            uttr_input, _= self.adapter(frames_inputs, frames_lengths)
        
        # build local intra context graph
        # dialog_lengths
        # self.cache_graph = 
        graph_key = ",".join([str(int(length_i)) for length_i in frames_lengths])
        edges_list = self.cache_graph.get(graph_key, build_intra_graph(frames_lengths, to_future_link=self.cfg.causalGNN.context_after, to_past_link=self.cfg.causalGNN.context_before, device=frames_inputs.device))
        if graph_key not in self.cache_graph.keys():
            self.cache_graph[graph_key] = edges_list
            
        causal_inputs = torch.cat([frames_inputs[i, :length_i] for i,length_i in enumerate(frames_lengths)])
        
        batches = torch.cat([torch.LongTensor([i]*length_i) for i, length_i in enumerate(frames_lengths)], 0).to(frames_inputs.device)
        xc, xo, xco, represent = self.causalGNN(causal_inputs, edges_list, batches, None)
        graph_adj = build_intra_graph(dialog_lengths, 
            to_future_link=self.cfg.dialogModel.to_future_link, 
            to_past_link=self.cfg.dialogModel.to_past_link, 
            device = frames_inputs.device)
        if self.cfg.get("residual", True):
            represent = represent + self.residual(uttr_input)
        x, res = self.dialogModel(represent, graph_adj)
        log.update(res)
            # , xc, xco
        return [x, xo, xc, xco], log
    def loss(self, outputs, labels, **kward):
        dia_out, o_logs, c_logs, co_logs = outputs
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(labels.device) / len(self.cfg.classes_name)
        o_loss = F.cross_entropy(o_logs, labels)
        c_loss = F.kl_div(F.softmax(c_logs, -1), uniform_target, reduction='batchmean')
        co_loss = F.cross_entropy(co_logs, labels)
        dialog_loss = F.cross_entropy(dia_out, labels)
        loss = dialog_loss + self.cfg.causalGNN.args.c * c_loss + self.cfg.causalGNN.args.o * o_loss + self.cfg.causalGNN.args.co * co_loss
        return loss 

class CRGNN(MyModel):
    def __init__(self, cfg) -> None:
        super().__init__()
        # cal model
        self.cfg = cfg
        causalGNN_input_dim = cfg.uttr_input_dim
        if cfg.get("adapter", None) != None:
            self.adapter = AdapterModel(input_dim = cfg.uttr_input_dim, **cfg.adapter) # self-attention
            causalGNN_input_dim = self.cfg.adapter.output_dim
        self.causalGNN = \
            getattr(causalGNNs, 
                    cfg.causalGNN.name)\
                        (num_features = causalGNN_input_dim,
                         num_classes = len(cfg.classes_name), 
                         **cfg.causalGNN)     # context
                        
        if self.cfg.get("residual", True):
            self.residual = nn.Sequential(
                nn.Linear(causalGNN_input_dim, cfg.causalGNN.args.hidden),
                nn.ReLU()
            )
            
            
        # dialog model
        self.dialogModel = \
            getattr(dialogModels, 
                    cfg.dialogModel.name)\
                        (
                            input_size = cfg.causalGNN.args.hidden,
                            out_size = len(cfg.classes_name),
                            **cfg.dialogModel)   # adgcn
        
    def forward(self, frames_inputs=None, frames_lengths=None, uttr_input=None, dialog_lengths=None, is_eval=False, epoch=0,**kwards):
        log = {}
        if self.cfg.get("adapter", None) != None:
            uttr_input, _= self.adapter(frames_inputs, frames_lengths)
        # build local context graph
        node_indexes, edges_list, batches, keypoints = build_context_graph(dialog_lengths, context_before=self.cfg.causalGNN.context_before, context_after=self.cfg.causalGNN.context_after, device=uttr_input.device)
        
        xc, xo, represent = self.causalGNN(uttr_input[node_indexes], edges_list, batches, keypoints)
        graph_adj = build_intra_graph(dialog_lengths, 
            to_future_link=self.cfg.dialogModel.to_future_link, 
            to_past_link=self.cfg.dialogModel.to_past_link, 
            device = uttr_input.device)
        # if epoch > self.cfg.get("freeze_epoch", 1e6):
        #     xo = xo.detach()
        #     xc = xc.detach()
        #     represent = represent.detach()
        if self.cfg.get("residual", True):
            represent = represent + self.residual(uttr_input)
        x, res = self.dialogModel(represent, graph_adj)
        log.update(res)

            # , xc, xco
        return [x, xo, xc], log
    def loss(self, outputs, labels, **kward):
        dia_out, o_logs, c_logs = outputs
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(labels.device) / len(self.cfg.classes_name)
        o_loss = super().loss(o_logs, labels)
        c_loss = F.kl_div(F.softmax(c_logs, -1), uniform_target, reduction='batchmean')
        # co_loss = super().loss(co_logs, labels)
        dialog_loss = super().loss(dia_out, labels)
        loss = dialog_loss + self.cfg.causalGNN.args.c * c_loss + self.cfg.causalGNN.args.o * o_loss 
        # + self.cfg.causalGNN.args.co * co_loss
        return loss  
    
# class CoAttention(nn.Module):
#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.norm0 = nn.LayerNorm(cfg.causalGNN.args.hidden)
#         self.norm1 = nn.LayerNorm(cfg.dialogModel.out_size)
#         self.co_attention = nn.Sequential(
#             nn.Linear(cfg.causalGNN.args.hidden+cfg.dialogModel.out_size, 2, bias=False),
#             nn.Softmax(-1))
#         self.out_size = cfg.causalGNN.args.hidden

#     def forward(self, x1, x2, **kward):
#         x1 = self.norm0(x1)
#         x2 = self.norm1(x2)
#         attention_weight = self.co_attention(torch.concat([x1, x2], -1))
#         return x1 * attention_weight[:,0].unsqueeze(-1) + x2 * attention_weight[:,1].unsqueeze(-1)
    
class CoAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.norm0 = nn.LayerNorm(cfg.causalGNN.args.hidden)
        self.norm1 = nn.LayerNorm(cfg.dialogModel.out_size)
        self.co_attention = nn.Sequential(
            nn.Linear(cfg.causalGNN.args.hidden+cfg.dialogModel.out_size, 2, bias=False),
            nn.Softmax(-1))
        self.out_size = cfg.causalGNN.args.hidden

    def forward(self, x1, x2, **kward):
        x1 = self.norm0(x1)
        x2 = self.norm1(x2)
        attention_weight = self.co_attention(torch.concat([x1, x2], -1))
        return x1 * attention_weight[:,0].unsqueeze(-1) + x2 * attention_weight[:,1].unsqueeze(-1)
    
class PureConcat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm0 = nn.LayerNorm(cfg.causalGNN.args.hidden)
        self.norm1 = nn.LayerNorm(cfg.dialogModel.out_size)
        self.out_size = cfg.dialogModel.out_size + cfg.causalGNN.args.hidden

    def forward(self, x1, x2, **kward):
        x1 = self.norm0(x1)
        x2 = self.norm1(x2)
        return torch.concat([x1, x2], -1)
    
class MultiheadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm0 = nn.LayerNorm(cfg.causalGNN.args.hidden)
        self.norm1 = nn.LayerNorm(cfg.dialogModel.out_size)
        self.cross_attention_fusion_ld = nn.MultiheadAttention(cfg.dialogModel.out_size, cfg.featureFusion.cross_attention_heads, dropout=cfg.featureFusion.get("dropout_cs", 0.5))
        self.cross_attention_fusion_dl = nn.MultiheadAttention(cfg.causalGNN.args.hidden, cfg.featureFusion.cross_attention_heads, dropout=cfg.featureFusion.get("dropout_cs", 0.5))
        # return cfg.dialogModel.out_size + cfg.causalGNN.args.hidden
        self.out_size = cfg.dialogModel.out_size + cfg.causalGNN.args.hidden

    def forward(self, x1, x2, dialog_lengths):
        x1 = self.norm0(x1)
        x2 = self.norm1(x2)
        feats_local, feats_dialog = self._pad_to_dialog(x1, x2,  dialog_lengths)
        key_padding_mask = torch.ones(dialog_lengths.shape[0], feats_local.shape[0], dtype=int).to(feats_local.device)
        for i, uttrs_num in enumerate(dialog_lengths):
            key_padding_mask[i, : uttrs_num] = 0
        key_padding_mask = key_padding_mask.to(torch.bool)
        feats_local_cs, _ = self.cross_attention_fusion_ld(feats_local, feats_dialog, feats_dialog, key_padding_mask)
        feats_dialog_cs, _ = self.cross_attention_fusion_dl(feats_dialog, feats_local, feats_local, key_padding_mask)
        feats_local, feats_dialog = self._unpad_to_feats(feats_local_cs, feats_dialog_cs, dialog_lengths)
        return torch.concat([feats_local, feats_dialog], -1)
    def _unpad_to_feats(self, x1, x2, dialog_lengths):
        xx1 = []
        xx2 = []
        for i, uttrs_num in enumerate(dialog_lengths):
            # start = dialog_lengths[:i].sum()
            # end = start + dialog_lengths[i]
            xx1.append(x1[:uttrs_num,i,:])
            xx2.append(x2[:uttrs_num,i,:])

        xx1 = torch.cat(xx1)
        xx2 = torch.cat(xx2)
        return xx1, xx2
    def _pad_to_dialog(self, x1, x2, dialog_lengths):
        pad_x1 = []
        pad_x2 = []
        for i in range(len(dialog_lengths)):
            start = dialog_lengths[:i].sum()
            end = start + dialog_lengths[i]
            pad_x1.append(x1[start:end])
            pad_x2.append(x2[start:end])
        
        pad_x1 = pad_sequence(pad_x1)
        pad_x2 = pad_sequence(pad_x2)
        return pad_x1, pad_x2

class CausalLocalConcatDiaModel(MyModel):
    def __init__(self, cfg) -> None:
        super().__init__()
        # cal model
        self.cfg = cfg
        causalGNN_input_dim = cfg.uttr_input_dim
        if cfg.get("adapter", None) != None:
            self.adapter = AdapterModel(input_dim = cfg.uttr_input_dim, **cfg.adapter) # self-attention
            causalGNN_input_dim = self.cfg.adapter.output_dim
        self.causalGNN = \
            getattr(causalGNNs, 
                    cfg.causalGNN.name)\
                        (num_features = causalGNN_input_dim,
                         num_classes = len(cfg.classes_name), 
                         **cfg.causalGNN)     # context
        # dialog model
        self.dialogModel = \
            getattr(dialogModels, 
                    cfg.dialogModel.name)\
                        (
                            input_size = causalGNN_input_dim,
                            **cfg.dialogModel)   # adgcn
        self.attention_layer = eval(cfg.featureFusion.type)(cfg)

        self.featureFusion = nn.Sequential(
            nn.Linear(self.attention_layer.out_size, cfg.featureFusion.hidden),
            nn.Dropout(cfg.featureFusion.dropout),
            nn.ReLU(),
            nn.Linear(cfg.featureFusion.hidden, len(cfg.classes_name))
        )
        
        
    def forward(self, frames_inputs=None, frames_lengths=None, uttr_input=None, dialog_lengths=None, is_eval=False, epoch=0,**kwards):
        log = {}
        if self.cfg.get("adapter", None) != None:
            uttr_input, _= self.adapter(frames_inputs, frames_lengths)
        # build local context graph
        node_indexes, edges_list, batches, keypoints = build_context_graph(dialog_lengths, context_before=self.cfg.causalGNN.context_before, context_after=self.cfg.causalGNN.context_after, device=uttr_input.device)
        
        xc, xo, xco, represent = self.causalGNN(uttr_input[node_indexes], edges_list, batches,keypoints)
        graph_adj = build_intra_graph(dialog_lengths, 
            to_future_link=self.cfg.dialogModel.to_future_link, 
            to_past_link=self.cfg.dialogModel.to_past_link, 
            device = uttr_input.device)
        x, res = self.dialogModel(uttr_input, graph_adj)

        x = self.attention_layer(represent, x, dialog_lengths = dialog_lengths)
        x = self.featureFusion(x)

        log.update(res)
            # , xc, xco
        return [x, xo, xc, xco], log
    def loss(self, outputs, labels, **kward):
        dia_out, o_logs, c_logs, co_logs = outputs
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(labels.device) / len(self.cfg.classes_name)
        o_loss = super().loss(o_logs, labels)
        c_loss = F.kl_div(F.softmax(c_logs, -1), uniform_target, reduction='batchmean')
        co_loss = super().loss(co_logs, labels)
        dialog_loss = super().loss(dia_out, labels)
        loss = dialog_loss + self.cfg.causalGNN.args.c * c_loss + self.cfg.causalGNN.args.o * o_loss + self.cfg.causalGNN.args.co * co_loss
        return loss  

class CausalContextModel(MyModel):
    def __init__(self, cfg) -> None:
        super().__init__()
        # cal model
        self.cfg = cfg
        causalGNN_input_dim = cfg.uttr_input_dim
        if cfg.get("adapter", None) != None:
            self.adapter = AdapterModel(input_dim = cfg.uttr_input_dim, **cfg.adapter)
            causalGNN_input_dim = self.cfg.adapter.output_dim
        
        self.causalGNN = \
            getattr(causalGNNs, 
                    cfg.causalGNN.name)\
                        (num_features = causalGNN_input_dim + cfg.get("n_speakers", 0),
                         num_classes = len(cfg.classes_name), 
                         **cfg.causalGNN)
    def forward(self, frames_inputs, frames_lengths, uttr_input=None, dialog_lengths=None, is_eval=False, speakers=None, **kwards):
        log = {}
        if self.cfg.get("adapter", None) != None:
            uttr_input, _= self.adapter(frames_inputs, frames_lengths)
        # build local context graph
        node_indexes, edges_list, batches, keypoints = build_context_graph(dialog_lengths, context_before=self.cfg.causalGNN.context_before, context_after=self.cfg.causalGNN.context_after, device=uttr_input.device)
        # batches = [[i]*dialog_lengths[i] for i, ]
        if self.cfg.get("n_speakers", 0) != 0:
            speakers_con = torch.cat([F.one_hot(torch.FloatTensor(speakers_i).to(int).to(frames_inputs.device), self.cfg.n_speakers)
                            for speakers_i in speakers], 0)
            uttr_input = torch.cat([uttr_input, speakers_con], -1)
        xc, xo, xco, represtation = self.causalGNN(uttr_input[node_indexes], edges_list, batches, keypoints)
        return [xo, xo, xc, xco], log
        # return xo, res
    def loss(self, outputs, labels, **kward):

        o_logs, c_logs, co_logs = outputs[1:] # o: 有用， c:没用， co: 混合
        o_loss = F.cross_entropy(o_logs, labels) # 1
        
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(labels.device) / len(self.cfg.classes_name)
        c_loss = F.kl_div(F.softmax(c_logs, -1), uniform_target, reduction='batchmean') # 0.5
        # c_loss = -F.cross_entropy(c_logs, labels) # 0.5
        
        co_loss = F.cross_entropy(co_logs, labels) #
        
        loss = self.cfg.causalGNN.args.c * c_loss + self.cfg.causalGNN.args.o * o_loss + self.cfg.causalGNN.args.co * co_loss
        return loss  

class AdapterModel(nn.Module):
    def __init__(self, 
                 select,
                 input_dim,
                 projector_dim,
                 output_dim,
                 **cfg) -> None:
        super().__init__()
        self.projector = nn.Linear(input_dim, projector_dim)
        self.adapter = eval(select)(
            input_dim = projector_dim,
            output_dim = output_dim,
            **cfg.get(select)
            )
    def forward(self, frames_inputs, frames_lengths, **kwards):
        res = {}
        out, _= self.adapter(self.projector(frames_inputs), frames_lengths)
        return out, res
        

class IntraModel(MyModel):
    def __init__(self, cfg) -> None:
        super().__init__()
        # cal model
        self.cfg = cfg
        self.adapter =AdapterModel(input_dim = cfg.uttr_input_dim, output_dim= len(self.cfg.classes_name), **cfg.adapter)
    def forward(self, frames_inputs, frames_lengths, **kwards):
        res = {}
        out, _= self.adapter(frames_inputs, frames_lengths)
        return out, res

class DSTModel(MyModel):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.projector = nn.Linear(cfg.uttr_input_dim, cfg.adapter.projector_dim)
        self.adapter=nn.Sequential(
            # nn.Linear(cfg.uttr_input_dim, cfg.adapter.projector_dim),
            nn.ReLU(),
            nn.Dropout(cfg.adapter.dropout),
            nn.Conv1d( cfg.adapter.projector_dim, cfg.adapter.hidden_sizes[0], cfg.adapter.kernel_sizes[0]),
            nn.ReLU(),
            nn.Dropout(cfg.adapter.dropout),
            nn.Conv1d( cfg.adapter.hidden_sizes[0], cfg.adapter.hidden_sizes[1], cfg.adapter.kernel_sizes[1]),
            nn.AdaptiveAvgPool1d(self.cfg.dst.length)
        )
        self.cfg.dst.update(
            input_dim = cfg.adapter.hidden_sizes[-1],
            num_classes = len(cfg.classes_name)
        )
        self.dst = DST(**self.cfg.dst)
    def forward(self, frames_inputs, frames_lengths, **kwards):
        res = {}
        out = self.adapter(self.projector(frames_inputs).permute(0,2,1)).permute(0,2,1)
        out = self.dst(out)
        return out, res

class DWFormerModel(MyModel):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.dwFormer.update(
            feadim = cfg.uttr_input_dim,
            classnum = len(cfg.classes_name)
        )
        self.dwFormer = DWFormer(**self.cfg.dwFormer)
    def forward(self, frames_inputs, frames_lengths, **kwards):
        res = {}
        # x = self.projector(frames_inputs)
        mask = torch.ones(frames_inputs.shape[:2], device=frames_inputs.device)
        
        for i, frame_length in enumerate(frames_lengths):
            mask[i, :frame_length] = 0
        out = self.dwFormer(frames_inputs, mask)
        return out, res
    
class LabelAdaptiveMixupModel(MyModel):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.projector = nn.Linear(cfg.uttr_input_dim, cfg.adapter.projector_dim)
        
        self.cfg.labelAdaptiveMixup.update(
            input_dim = cfg.adapter.projector_dim,
            num_classes = len(cfg.classes_name)
        )
        self.labelAdaptiveMixup_fc1 = nn.Linear(cfg.adapter.projector_dim, self.cfg.labelAdaptiveMixup.featDim)
        self.labelAdaptiveMixup_fc2 = nn.Sequential(
            torch.nn.Linear(self.cfg.labelAdaptiveMixup.featDim, self.cfg.labelAdaptiveMixup.num_classes),
            
        )
        self.sm = nn.LogSoftmax(-1)
        self.center_crit=CenterLoss(num_classes=self.cfg.labelAdaptiveMixup.num_classes, 
                                   feat_dim=self.cfg.labelAdaptiveMixup.featDim)
        self.kl_crit = nn.KLDivLoss(reduction='batchmean')
    def forward(self, frames_inputs, **kwards):
        res = {}
        x = self.projector(frames_inputs[:,0,:])
        fea = self.labelAdaptiveMixup_fc1(x)
        out = self.labelAdaptiveMixup_fc2(fea)
        return [out, fea], res
    def loss(self, outputs, labels, **kwards):
        res_emo, feat_emo  = outputs
        loss_res = self.kl_crit(self.sm(res_emo), F.one_hot(labels, self.cfg.labelAdaptiveMixup.num_classes).float().to(labels.device))
        loss_cen = self.center_crit(feat_emo, labels)
        loss = loss_res + self.cfg.labelAdaptiveMixup.alpha * loss_cen
        return loss

class CNNMfcc(MyModel):
    def __init__(self, cfg) -> None:
        super().__init__()
        # cal model
        self.cfg = cfg
        self.tfcnn = TFCNN(1, 128, 128, 0.5, cfg.input_size)
        
        
        self.classify = nn.Sequential(
            nn.Linear(128, len(cfg.classes_name))
        )
    def forward(self, uttr_input=None, frames_inputs=None, **kwards):
        x, _ = self.tfcnn(frames_inputs.permute(0,2,1))
        x = self.classify(x)
        return x, {} 
           
class MLPModel(MyModel):
    def __init__(self, cfg) -> None:
        super().__init__()
        # cal model
        self.cfg = cfg
        self.MLP = nn.Sequential(
            nn.LayerNorm(cfg.uttr_input_dim),
            nn.Linear(cfg.uttr_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(cfg.classes_name))
        )
    def forward(self, uttr_input=None, frames_inputs=None, **kwards):
        if uttr_input==None:
            uttr_input  = frames_inputs[:,0,:]
        return self.MLP(uttr_input), {}  


              



if __name__ == "__main__":
    from myDatasets import IEMOCAPDataset
    from torch.utils.data import DataLoader
    my_dataset = IEMOCAPDataset("train", data_path="/home/users/ntu/n2107167/lnespnet/chengqi/ADGCNForEMC/features/IEMOCAP_features/features/f/cross_val5/features/no_train_audio_features.pkl")
    my_dataset_valid = IEMOCAPDataset("valid", data_path="/home/users/ntu/n2107167/lnespnet/chengqi/ADGCNForEMC/features/IEMOCAP_features/features/f/cross_val5/features/no_train_audio_features.pkl")
    train_dataloader = DataLoader(my_dataset,
                            batch_size=10,
                            sampler=None,
                            collate_fn=my_dataset.collate_fn,
                            num_workers=1,
                            pin_memory=False)
    valid_dataloader = DataLoader(my_dataset_valid,
                            batch_size=10,
                            sampler=None,
                            collate_fn=my_dataset_valid.collate_fn,
                            num_workers=1,
                            pin_memory=False)
    model = DialogModel()                        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.000001)
    
    for i in range(200):
        for data in train_dataloader:
            optimizer.zero_grad(),
            out, res = model(**data)
            loss = model.loss(out, data["label"])
            loss.backward()
            optimizer.step()
        # print("%dth epoch: train loss %4f"%(i, loss))
        val_out =[]
        val_label =[]
        with torch.no_grad():
            for data in valid_dataloader:
                out, res = model(**data)
                val_out.append(out)
                val_label.append(data["label"])
            log = model.merits(torch.cat(val_out), torch.cat(val_label))
        print("%dth epoch: val acc %4f"%(i, log["accuracy"]))
