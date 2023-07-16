from transformers import Data2VecAudioModel, Data2VecTextModel, Data2VecVisionModel, Wav2Vec2Model
import torch
import torch.nn.functional as F
from torch import nn, no_grad
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, GCN2Conv, GCNConv, APPNP
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
# from src.ADGCN import ADGCNForDialog
import torchvision
import src.CausalGCN as causalGNNs
import src.dialogModels as dialogModels
from src.dialogModels import *
import numpy as np
from src.emotionFroms3pl import *
from src.tfcnn import *
from src.dialogGCN import DialogueGCNModel

class MyModel(nn.Module):
    def loss(self, classes, labels, mode="classification"):
        # return F.cross_entropy(classes, labels, weight=torch.tensor([1.0,2.0,2.0,3.0]).to(classes.device))
        if mode == "classification":
            if self.cfg.get("used_focal_loss", False):
                return torchvision.ops.sigmoid_focal_loss(classes, F.one_hot(labels, len(self.cfg.classes_name)).float().to(labels.device), reduction="mean")
            return F.cross_entropy(classes, labels)
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
    
from transformers import AutoModel

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
    
class DialogGCN(MyModel):
    def __init__(self, cfg, **kwards):
        super().__init__()
        self.cfg = cfg
        self.cfg.dialogModel["D_m"] = cfg.uttr_input_dim
        self.cfg.dialogModel["n_classes"] = len(cfg.classes_name)
        if cfg.get("adapter", None) != None:
            self.adapter = AdapterModel(input_dim = cfg.uttr_input_dim, **cfg.adapter)
        self.cfg.dialogModel["D_m"] = self.cfg.adapter.output_dim
        self.dialogGCN = DialogueGCNModel(**self.cfg.dialogModel)

    
    def forward(self,  frames_inputs=None, frames_lengths=None, uttr_input=None, dialog_lengths=None, speakers = None, **kwards):
        res = {}
        # extract features
        if self.cfg.get("adapter", None) != None:
            uttr_input, _= self.adapter(frames_inputs, frames_lengths)
        
        rnn_inputs = torch.split(uttr_input, dialog_lengths.tolist())
        U = pad_sequence(rnn_inputs)
        with no_grad():
            speakers = torch.stack([speakers, 1-speakers], -1)
            speakers = torch.split(speakers, dialog_lengths.tolist())

            qmask = pad_sequence(speakers)
            umask = torch.zeros([qmask.shape[1], qmask.shape[0]]).to(uttr_input.device)
            for i,dialog_length in enumerate(dialog_lengths): 
                
                umask[i, :dialog_length] = 1
        
        log_prob, e_i, e_n, e_t, e_l = self.dialogGCN(U, qmask, umask, dialog_lengths.tolist())
        res.update(e_i=e_i, e_n=e_n, e_t=e_t, e_l=e_l)
        return log_prob, res

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
        xc, xo, xco, represent = self.causalGNN(causal_inputs, edges_list, batches, None, not (is_eval and not self.cfg.causalGNN.args.eval_random) and self.cfg.causalGNN.args.with_random )
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

class CausalDiaModel(MyModel):
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
                        (num_features = causalGNN_input_dim,
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
        
    def forward(self, frames_inputs=None, frames_lengths=None, uttr_input=None, dialog_lengths=None, is_eval=False, epoch=0,**kwards):
        log = {}
        if self.cfg.get("adapter", None) != None:
            uttr_input, _= self.adapter(frames_inputs, frames_lengths)
        # build local context graph
        node_indexes, edges_list, batches, keypoints = build_context_graph(dialog_lengths, context_before=self.cfg.causalGNN.context_before, context_after=self.cfg.causalGNN.context_after, device=uttr_input.device)
        
        xc, xo, xco, represent = self.causalGNN(uttr_input[node_indexes], edges_list, batches,keypoints, not (is_eval and not self.cfg.causalGNN.args.eval_random) and self.cfg.causalGNN.args.with_random )
        graph_adj = build_intra_graph(dialog_lengths, 
            to_future_link=self.cfg.dialogModel.to_future_link, 
            to_past_link=self.cfg.dialogModel.to_past_link, 
            device = uttr_input.device)
        if epoch > self.cfg.get("freeze_epoch", 1e6):
            xo = xo.detach()
            xc = xo.detach()
            xco = xo.detach()
            represent = represent.detach()
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
                        (num_features = causalGNN_input_dim,
                         num_classes = len(cfg.classes_name), 
                         **cfg.causalGNN)
    def forward(self, frames_inputs, frames_lengths, uttr_input=None, dialog_lengths=None, is_eval=False, **kwards):
        log = {}
        if self.cfg.get("adapter", None) != None:
            uttr_input, _= self.adapter(frames_inputs, frames_lengths)
        # build local context graph
        node_indexes, edges_list, batches, keypoints = build_context_graph(dialog_lengths, context_before=self.cfg.causalGNN.context_before, context_after=self.cfg.causalGNN.context_after, device=uttr_input.device)
        # batches = [[i]*dialog_lengths[i] for i, ]
        xc, xo, xco, represtation = self.causalGNN(uttr_input[node_indexes], edges_list, batches,keypoints, not (is_eval and not self.cfg.causalGNN.args.eval_random) and self.cfg.causalGNN.args.with_random )
        return [xo, xc,  xco], log
        # return xo, res
    def loss(self, outputs, labels, **kward):

        o_logs, c_logs, co_logs = outputs # o: 有用， c:没用， co: 混合
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
        # self.causalGNN = \
        #     getattr(causalGNNs, 
        #             cfg.causalGNN.name)\
        #                 (num_features = cfg.uttr_input_dim,
        #                  num_classes = len(cfg.classes_name), 
        #                  **cfg.causalGNN)
    def forward(self, uttr_input=None, frames_inputs=None, **kwards):
        if uttr_input==None:
            uttr_input  = frames_inputs[:,0]
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
