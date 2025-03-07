from omegaconf import OmegaConf

from experiment import Experiment
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support

import torch
class InferenceExperiment(Experiment):
    def __init__(self, cfg):
        super().__init__(cfg)
    # def on_test_start(self) -> None:
    #     super().on_train_start()
    #     return super().on_test_start()
    def test_step(self, batch, batch_idx):
            batch.update(is_eval=True)
            out, res = self(batch)
            labels = batch["labels"]
            loss = self.loss(
                out,
                labels,
                mode = self.cfg.dataset.get("train_mode", "classification")
            )
            log = {}
            if type(out)==type([]):
                log["out"] = out[0]
            else:
                log["out"] = out
            log["labels"] = labels
            log["loss"] = loss
            log["shiftIndexes"] = torch.LongTensor(self.shiftByDialog(batch))
            log["shiftIndexesSpe"] = torch.LongTensor(self.shiftBySpeaker(batch))
            # log["shiftIndexes"] = self.shiftBySpeaker(batch)
            if self.cfg.get("show_tsne", False):
                log.update(res)
            return log
    def test_epoch_end(self, outputs):
        aggra_out = torch.cat([x["out"] for x in outputs])
        aggra_labels = torch.cat([x["labels"] for x in outputs])
        test_loss = torch.stack([x["loss"] for x in outputs]).mean().data
        
        shift_labels = torch.cat([x["labels"][x["shiftIndexes"]] for x in outputs])
        shift_outs = torch.cat([x["out"][x["shiftIndexes"]] for x in outputs])

        shift_labelsSpe = torch.cat([x["labels"][x["shiftIndexesSpe"]] for x in outputs])
        shift_outsSpe = torch.cat([x["out"][x["shiftIndexesSpe"]] for x in outputs])

        shift_logSpe = accuracy_score(shift_outsSpe.data.cpu().numpy().argmax(-1), shift_labelsSpe.cpu().numpy())
        shift_log = accuracy_score(shift_outs.data.cpu().numpy().argmax(-1), shift_labels.cpu().numpy())
        
        print(f"dia | {(1-shift_log)*len(shift_labels)} | {len(shift_labels)} | {1-shift_log} |")
        print(f"Spe | {(1-shift_logSpe)*len(shift_labelsSpe)} | {len(shift_labelsSpe)} | {1-shift_logSpe} |")

        # test_loss = self.loss(aggra_out, aggra_labels, self.cfg.dataset.get("train_mode", "classification"))
        log = self.metric(aggra_out, aggra_labels)
        
            
        self.log("test_loss", float(test_loss), on_epoch=True, on_step =False)
        self.log("test_accuracy", float(log["accuracy"]), on_epoch=True, on_step =False)
        # self.log_dict(log)
        # if self.my_logger != None:
        print("epoch%d:val_wacc=%.4f, val_uacc=%.4f, val_single_acc=%s val_f1-macro=%.4f, val_f1-micro=%.4f,val_f1-weighted=%.4f, val_loss=%.4f;"%(self.current_epoch,log["accuracy"],  log["unweight_accuracy"],log["single_accuracy"],log["f1_macro"], log["f1_micro"],log["f1_weighted"], float(test_loss)))

    def shiftByDialog(self, batch):
        shiftIndexes = []
        for dia_i in range(len(batch["dialog_lengths"])):
            last_label = -1
            for index, label in enumerate(batch["labels"][sum(batch["dialog_lengths"][:dia_i]):sum(batch["dialog_lengths"][:(dia_i+1)])]):
                if last_label != -1 and label != last_label:
                    shiftIndexes.append(index + sum(batch["dialog_lengths"][:dia_i]))
                last_label = label
        return list(map(int, shiftIndexes))
    
    def shiftBySpeaker(self, batch):
        shiftIndexes = []
        for dia_i in range(len(batch["dialog_lengths"])):
            # last_label = -1
            speakers_last_label = {}
            for index, label in enumerate(batch["labels"][sum(batch["dialog_lengths"][:dia_i]):sum(batch["dialog_lengths"][:(dia_i+1)])]):
                if batch["speakers"][dia_i][index] in speakers_last_label.keys() and label.item() != speakers_last_label[batch["speakers"][dia_i][index]]:
                    shiftIndexes.append(index + sum(batch["dialog_lengths"][:dia_i]))
                speakers_last_label[batch["speakers"][dia_i][index]] = label.item()
        return list(map(int, shiftIndexes))
        
def test(cfg):
    experiment = InferenceExperiment(cfg)
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoints)
    earlystop_callback = EarlyStopping(**cfg.earlystop)
    trainer = Trainer(**cfg.trainer, callbacks=[checkpoint_callback, earlystop_callback])
    # trainer.fit(experiment, ckpt_path=cfg.trainer.resume_from_checkpoint)
    
    trainer.test(experiment, ckpt_path=cfg.trainer.resume_from_checkpoint)


if __name__=="__main__":
    result_dir = f"best_results/iemocap/results/Wavlm_DialogGCN/flod5"
    best_ckpt = "epoch=48-val_loss=1.70776-val_accuracy=0.76632"
    config_path = result_dir + "/hparams.yaml"
    cfg = OmegaConf.load(config_path).cfg
    # cfg.checkpoints["dirpath"] = result_dir + "/checkpoints"
    cfg.trainer["resume_from_checkpoint"] = result_dir + f"/checkpoints/{best_ckpt}.ckpt"
    seed_everything(cfg.seed)
    test(cfg)