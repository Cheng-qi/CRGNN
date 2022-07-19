import json
import pytorch_lightning as pl
import gc
import math
from abc import ABC
import seaborn as sns
import matplotlib.pyplot as plt
import sys 
import yaml
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch.optim.adam import Adam
from torch.optim import Adamax, Adadelta
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from units import plot_confusion_matrix, parse_args

from myDatasets import *
from model import *
from pytorch_lightning import Trainer, seed_everything

class Experiment(LightningModule, ABC):
    def __init__(self, cfg):
        super().__init__()
        # self.hparams = OmegaConf.to_container(cfg, resolve=True)
        self.cfg = cfg
        cfg.dataset.update(pretrain_model=cfg.model.pretrain_model)
        self.model = DialogModel(**cfg.model)
        # self.model = SimpleLinear(**cfg.model)

    def forward(self, data):
        return self.model(**data)

    def prepare_data(self):
        if self.cfg.model.pretrain_model == None:
            IEMOCAPDataset = IEMOCAPDataset
        else:
            IEMOCAPDataset = IEMOCAPAudioDatasetsRaw

        self.train_dataset = IEMOCAPDataset(split = "train", **self.cfg.dataset)
        self.val_dataset = IEMOCAPDataset(split = "valid", **self.cfg.dataset)
        self.test_dataset = IEMOCAPDataset(split = "test", **self.cfg.dataset)

    def configure_optimizers(self):
        lr = self.cfg.optimizer.learning_rate
        weight_decay = self.cfg.optimizer.weight_decay
        eps = 1e-2 / float(self.cfg.data_loader.batch_size) ** 2
        if self.cfg.optimizer.type.lower() == "rmsprop":
            optimizer = RMSprop(self.parameters(),
                                lr=lr,
                                momentum=self.cfg.optimizer.momentum,
                                eps=eps,
                                weight_decay=weight_decay)
        elif self.cfg.optimizer.type.lower() == "adam":
            optimizer = Adam(self.parameters(),
                             lr=lr,
                            #  eps=eps,
                             weight_decay=weight_decay)
        elif self.cfg.optimizer.type.lower() == "adamax":
            optimizer = Adamax(self.parameters(),
                             lr=lr,
                            #  eps=eps,
                             weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer type.")


        if not self.cfg.lr_scheduler.active:
            return optimizer
        scheduler = ExponentialLR(optimizer=optimizer,
                                  gamma=self.cfg.lr_scheduler.decay_rate)

        return [optimizer], [scheduler]

    def train_dataloader(self):

        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.data_loader.batch_size,
                          collate_fn=self.train_dataset.collate_fn,
                          num_workers=self.cfg.data_loader.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.data_loader.batch_size,
                           collate_fn=self.val_dataset.collate_fn,
                          num_workers=self.cfg.data_loader.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.data_loader.batch_size,
                          collate_fn=self.val_dataset.collate_fn,
                          num_workers=self.cfg.data_loader.num_workers)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def on_epoch_start(self):
        torch.cuda.empty_cache()
        if not self.cfg.lr_scheduler.active:
            return

        current_lr = self.get_lr(self.trainer.optimizers[0])
        self.logger.experiment.add_scalar(
            'learning_rate', current_lr, self.current_epoch)

    def on_batch_end(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()

    def training_step(self, batch, batch_idx):
        out, _ = self(batch)
        labels = batch["label"]
        loss = self.model.loss(
            out,
            labels
        )
        log = self.model.merits(out, labels)
        # log["train_loss"] = loss.detach()
        self.log("train_accuracy", log["accuracy"])
        self.log("train_loss", loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        out, res = self(batch)
        labels = batch["label"]
        loss = self.model.loss(
            out,
            labels
        )
        log = self.model.merits(out, labels)
        log["out"] = out
        log["labels"] = labels
        log["loss"] = loss

        # self.log
        return log

    def validation_epoch_end(self, outputs):
        aggra_out = torch.cat([x["out"] for x in outputs])
        aggra_labels = torch.cat([x["labels"] for x in outputs])
        val_loss = self.model.loss(aggra_out, aggra_labels)
        log = self.model.merits(aggra_out, aggra_labels)
        c_m = confusion_matrix(aggra_labels.data.cpu().numpy(), aggra_out.data.cpu().argmax(-1).numpy())
        if self.cfg.model.n_classes == 4:
            classes = ["N", "A", "S", "H"]
        else:
            classes = ["hap", "sad", "neu", "ang", "exc", "fru"]
        fig = plot_confusion_matrix(c_m, normalize= True, classes=classes)

        self.logger.experiment.add_figure("conf/test_nor", fig, self.current_epoch)
        
        # self.log_dict()
        # fig = plt.figure()
        # x_tsne = tsne.fit_transform(test_embeddings)
        # test_label_str = [["N", "A", "S", "H"][i] for i in test_label.tolist()]
        # sns.scatterplot(x=x_tsne[:,0],y=x_tsne[:,1],hue=test_label_str)
        # writer.add_figure("tsne/test", fig, e)
        self.log("val_loss", float(val_loss), on_epoch=True, on_step =False)
        self.log("val_accuracy", float(log["accuracy"]), on_epoch=True, on_step =False)
        # log["val_accuracy"] = log["accuracy"]
        # log["val_accuracy"] = log["accuracy"]
        # log["val_accuracy"] = log["accuracy"]

        # return {'val_loss': val_loss, 'log': log}
        # return {'val_loss': val_loss, 'log': log}

    def test_step(self, batch, batch_idx):
        loss =0
        accuracy=0
        return {'test_loss': loss, 'accuracy': accuracy}
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        log = {'test_loss': avg_loss, 'test_accuracy': avg_acc}
        return {'test_loss': avg_loss, 'log': log}
def train(cfg):
    
    experiment = Experiment(cfg)
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoints)
    earlystop_callback = EarlyStopping(**cfg.earlystop)
    trainer = Trainer(**cfg.trainer, callbacks=[checkpoint_callback, earlystop_callback])
    trainer.fit(experiment)
if __name__=="__main__":
    config_path = "config.yaml"
    cfg = OmegaConf.load(config_path)
    if len(sys.argv) > 1:
        overwrite = json.loads(sys.argv[-1])
        for k1, v1 in overwrite.items():
            for k2, v2 in v1.items():
                cfg.get(k1)[k2] = v2
    seed_everything(cfg.seed)
    train(cfg)