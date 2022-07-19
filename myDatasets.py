from matplotlib import transforms
from matplotlib.pyplot import axis
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
import numpy as np
import soundfile as sf
from units import *
class IEMOCAPAudioDatasetsRaw(RawAudioDataset):
    def __init__(
        self, 
        data_path,
        video_path = "/home/projects/12001458/chengqi/huggingface/mydatasets/iemocap_original/IEMOCAP/{session_id}/sentences/wav/{video}/{segment}.wav",
        split="train", 
        valid_session=5,
        max_sample_size=16000, 
        min_sample_size=5000, 
        max_tokens=1500000,
        sample_rate=16000, 
        shuffle=True, 
        pad=True, 
        normalize=False, 
        pretrain_model = "d2v",
        extract_feature = False,
        compute_mask_indices = False,
        text_compression_level=..., 
        **mask_compute_kwargs):
        super().__init__(sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs)
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.normalize = normalize
        self.sample_rate = sample_rate

        self.video_path = video_path

                # data process
        with torch.no_grad():
            if pretrain_model == "d2v":
                self.data_processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
            # self.data_processor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
        self.extract_feature = extract_feature

        self.max_tokens = max_tokens
        print("load %s"%data_path)

        data_pd = pd.read_csv(data_path, delimiter=",")
        if split=="train":
            data_pd = data_pd[data_pd["session"].isin(["Session%d"%i for i in list(range(1,valid_session))+list(range(valid_session+1,6))])]
        elif split=="valid":
            data_pd = data_pd[data_pd["session"]=="Session%d"%valid_session]
        self.dialog_ids = []
        self.label_gts = []
        self.sizes = []
        self.fnames = []

        self.fnames ={}
        self.label_gts ={}
        dialog_size = 0
        for i, row_i in data_pd.iterrows():
            # segment,speaker,label,text,video,session,dialog, num_frames = row_i
            segment,label,text,video,segment2,session,dialog,frames_num,wav_num = row_i
            # self.label_gts.append(int(label))
            if dialog not in self.fnames.keys():
                self.fnames[dialog] = []
                self.label_gts[dialog] = []
                self.dialog_ids.append(dialog)
                _ = self.sizes.append(dialog_size) if dialog_size>0 else 0
                dialog_size = 0
            self.label_gts[dialog].append(["N", "A", "S", "H"].index(label))
            # self.label_gts[dialog].append(label)
            # IEMOCAP/{session_id}/sestences/avi/{video}/{segment}.avi"
            fname = self.video_path.format(session_id=session,video=video, segment=segment)
            self.fnames[dialog].append(fname)
            dialog_size += max(min(wav_num, self.max_sample_size), self.min_sample_size)


        _ = self.sizes.append(dialog_size) if dialog_size>0 else 0

        self.sizes = np.array(self.sizes, dtype=np.int64)
        try:
            import pyarrow
            self.texts = pyarrow.array(self.texts)
        except:
            pass
        # self.set_bucket_info(num_buckets)
    def __len__(self):
        return len(self.dialog_ids)
        
    def __getitem__(self, index):
        dialog = self.dialog_ids[index]
        fnames = self.fnames[dialog]
        input_feats = []
        label_gts = []
        dialog_length = len(fnames)
        for i, fname_i in enumerate(fnames):
            wav, curr_sample_rate = sf.read(fname_i, dtype="float32")
            feats = torch.from_numpy(wav).float()
            if len(feats) < self.min_sample_size:
                with torch.no_grad():
                    feats = feats.repeat(round(self.min_sample_size/len(feats)+0.5))[:self.min_sample_size]
                    
            feats = self.postprocess(feats, curr_sample_rate)
            if len(feats) > self.max_sample_size:
                feats = self.crop_to_max_size(feats, self.max_sample_size)
            
            input_feats.append(feats)
            label_gts.append(self.label_gts[dialog][i])

        return {"id": index, 
        "source": input_feats, # (C, H, W)
        "label_gts":label_gts,
        "dialog_lengths":dialog_length
        }
    def batch_sampler(self):
        return MyBatchSampler(self.batch_by_size(self.sizes.argsort(), max_tokens=self.max_tokens))
    def collate_fn(self, samples):
        with torch.no_grad():
            return {
                "feats": self.data_processor([feats.numpy() for sample in samples for feats in sample["source"]], padding=self.pad, sampling_rate=self.sample_rate, return_tensors="pt"),
                "dialog_lengths":  torch.tensor([sample["dialog_lengths"] for sample in samples]),
                "labels": torch.tensor([label for sample in samples for label in sample["label_gts"]]),
            }




class IEMOCAPDataset(Dataset):
    def __init__(self, split="train", cross_val=5, data_path = '/home/projects/12001458/chengqi/conv-emotion/AudioDialogueGCN/IEMOCAP_features/features/f/cross_val5/features/epoch=155-val_loss=3.81424-val_accuracy=0.71152_audio_linear.pkl', **kwards):
        feature_path = data_path
        data = pickle.load(open(feature_path, 'rb'), encoding='latin1')
        print("load feature from %s"%feature_path)
        self.features, self.labels =  data["features"], data["labels"]
        self.session_ids = data["session_ids"]
        self.keys=[]
        if split == "train":
            for i in list(range(1,cross_val))+list(range(cross_val+1,6)):
                self.keys = self.keys+self.session_ids["Session%d"%i]
        elif split == "valid":
            self.keys = self.session_ids["Session%d"%cross_val]
        else:
            for i in list(range(1,6)):
                self.keys = self.keys+self.session_ids["Session%d"%i]

        self.len = len(self.keys)

    def __getitem__(self, index):
        dialog_id = self.keys[index]
        feats = torch.FloatTensor(self.features[dialog_id])
        dialog_length = len(self.labels[dialog_id])
        label = torch.LongTensor(self.labels[dialog_id])
        return feats, dialog_length, label

    def __len__(self):
        return self.len
    def collate_fn(self, data):
        feats = []
        dialog_lengths = []
        labels = []
        for data_i in data:
            feats.append(data_i[0])
            dialog_lengths.append(data_i[1])
            labels.extend(data_i[2])
        return {"feats": torch.cat(feats, axis=0), "dialog_lengths":torch.LongTensor(dialog_lengths), "label":torch.LongTensor(labels)} # feats, dialog_length, label
  

if __name__ == "__main__":
    # video_dataset = VideoIEMOCAPDataset(True)
    # audio_dataset = AudioIEMOCAPDataset(True)
    # data_0 = video_dataset[0]
    dataset = IEMOCAPAudioDatasetsRaw("/home/users/ntu/n2107167/lnespnet/chengqi/ADGCNForEMC/datasets/iemocap_original/audio/IEMOCAP_Audio_4.csv",split="test")
    
    
    # dataset = IEMOCAPDataset("train")
    dataloader = DataLoader(dataset,
                            batch_sampler=dataset.batch_sampler(),
                            collate_fn=dataset.collate_fn,
                            num_workers=1,
                            pin_memory=False)
    for data in dataloader:
        data