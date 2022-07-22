from transformers import Data2VecAudioModel, Data2VecTextModel, Data2VecVisionModel, Wav2Vec2Model
import torch
import torch.nn.functional as F
from torch import nn, no_grad
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, GCN2Conv, GCNConv, APPNP
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from ADGCN import ADGCNForDialog

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

class GATForDialog(torch.nn.Module):
    def __init__(self, num_features, head_num=3, hidden_size=64, n_classes=4, dropout=0.5) -> None:
        super(GATForDialog, self).__init__()
        self.head_num = head_num
        self.gconv1 = GATConv(num_features, hidden_size, self.head_num, dropout=dropout)
        self.act = nn.ReLU()
        self.gconv2 = GATConv(hidden_size*self.head_num, hidden_size, self.head_num, dropout=dropout)    
        self.gconv_out= GATConv(hidden_size*self.head_num, hidden_size, 1, concat=False, dropout=dropout)
        self.classfier = nn.Linear(hidden_size, n_classes)
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
    def __init__(self, input_size, hidden_size, layers=64, n_classes=4, dropout=0.6) -> None:
        super(GCNForDialog, self).__init__()
        self.gconvs = nn.ModuleList()
        self.gconvs.append(GCNConv(input_size, hidden_size))
        for i in range(layers-1):
            self.gconvs.append(GCNConv(hidden_size, hidden_size))
        self.classfier = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
    def forward(self, x, edge_index, **kwards):
        h = x
        for i, con in enumerate(self.gconvs):
            h = self.dropout(h)
            h = con(h, edge_index)
            h = self.act(h)
        out = self.classfier(h)
        return out

class GCNIIForDialog(torch.nn.Module):
    def __init__(self, input_size, hidden_size, layers=64, alpha=0.2, theta=0.5, n_classes=4, dropout=0.6) -> None:
        super(GCNIIForDialog, self).__init__()
        self.gconvs = nn.ModuleList()
        if input_size != hidden_size:
            self.project = nn.Linear(input_size, hidden_size)
        else:
            self.project = None
        for i in range(layers):
            self.gconvs.append(GCN2Conv(hidden_size, alpha, theta, i+1))
        self.classfier = nn.Linear(hidden_size, n_classes)
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
    def __init__(self, in_features, hidden_size, out_features, dropout=0.6) -> None:
        super(MLPForDialog, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_features)
        )
    def forward(self, x, **kwards):
        log = {}
        out = self.MLP(x)
        return out, log

class DialogModel(nn.Module):
    def __init__(self, 
                n_classes=4, 
                adaptive_size=2, 
                classModel="ADGCN",
                linkerModel_cnn1=[5, 2],
                linkerModel_cnn2=[3, 1],
                pretrain_model=None,
                graph_input_size=512,
                gat_head = 5,
                graph_hidden_size = 128,
                dropout=0.5,
                gnn_layers=10,
                alpha = 0.2,
                theta = 0.5,
                to_future_link = 4,
                to_past_link = 0,
                **kwards):
        super().__init__()
        self.to_future_link=to_future_link
        self.to_past_link=to_past_link


        self.pretrain_model = pretrain_model
        with torch.no_grad():
            if pretrain_model == "d2v":
                self.data2vecModel = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")
            elif pretrain_model == "w2v2":
                self.data2vecModel = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        if pretrain_model!=None:
            self.linkerModel = nn.Sequential(
                nn.Conv1d(768, 512, *linkerModel_cnn1),
                nn.GELU(),
                nn.Conv1d(512, 256, *linkerModel_cnn2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(adaptive_size),
                nn.Flatten(),
            )
        graph_input_size = adaptive_size * 256
        if classModel == "GAT":
            self.class_model = GATForDialog(graph_input_size, head_num=gat_head, hidden_size=graph_hidden_size, n_classes=n_classes, dropout=dropout)
        if classModel == "GCN":
            self.class_model = GCNForDialog(graph_input_size, layers=gnn_layers, n_classes=n_classes,dropout=dropout)
        elif classModel == "GCNII":
            assert graph_hidden_size==graph_hidden_size
            self.class_model = GCNIIForDialog(graph_input_size, hidden_size=graph_hidden_size, layers=gnn_layers, alpha=alpha, theta=theta, n_classes=n_classes, dropout=dropout)
        elif classModel == "ADGCN":
            assert graph_hidden_size==graph_hidden_size
            self.class_model = ADGCNForDialog(graph_input_size, nhidden=graph_hidden_size,  nlayers=gnn_layers, lamda=theta, n_classes=n_classes, dropout=dropout)
        elif classModel == "MLP":
            self.class_model = MLPForDialog(graph_input_size,graph_hidden_size, n_classes, dropout=dropout)
        else:
            raise("gnn type error")
    
    def forward(self, **data):
        res = {}
        dialog_lengths = data["dialog_lengths"]
        label = data.get("label", None)
        # extract features
        if self.pretrain_model != None:
            with torch.no_grad():
                x = self.data2vecModel(**data["feats"])
            x = x["last_hidden_state"].transpose(2,1)
            feats = self.linkerModel(x)
        else:
            feats = data["feats"]
        graph_adj = build_intra_graph(dialog_lengths, 
            to_future_link=self.to_future_link, 
            to_past_link=self.to_past_link, 
            device = feats.device)
        x, log = self.class_model(feats, graph_adj, **data)
        res.update(log)
        return x, res
    
    def loss(self, classes, labels):
        return F.cross_entropy(classes, labels)

    def merits(self, out, labels):
        out = out.data.cpu().numpy()
        pred_out = out.argmax(-1)
        labels = labels.cpu().numpy()
        accuracy = accuracy_score(labels, pred_out)
        f1_micro = f1_score(labels, pred_out, average="micro")
        f1_macro = f1_score(labels, pred_out, average="macro")
        # c_m = confusion_matrix(labels, pred_out)
        res = {
            "accuracy":accuracy, 
            "f1_micro":f1_micro, 
            "f1_macro":f1_macro, 
            # "c_m":torch.Tensor(c_m),
        }
        return res
    


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
