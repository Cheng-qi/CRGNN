from matplotlib import transforms
from matplotlib.pyplot import axis
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
import numpy as np
import soundfile as sf
from units import *
class IEMOCAP5531CrossValRaw(RawAudioDataset):
    def __init__(
        self, 
        split="train", 
        data_path ="data/iemocap_vad/IEMOCAP_4_5531.csv",
        # raw_path = "/home/projects/12001458/chengqi/huggingface/mydatasets/iemocap_original/IEMOCAP/{session_id}/sentences/wav/{video}/{segment}.wav",
        raw_path = "/data/ADGCNForEMC/data/iemocap_vad/vaddata/session{session_id}/{segment2}.wav",
        classes_name = ["N", "A", "S", "H"],
        cross_val=5,
        max_sample_size=320000, 
        min_sample_size=5000, 
        max_tokens=5770000,
        sample_rate=16000, 
        shuffle=True, 
        pad=True, 
        normalize=False, 
        model = "facebook/data2vec-audio-base-960h",
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

        self.raw_path = raw_path

                # data process
        with torch.no_grad():
            self.data_processor = AutoProcessor.from_pretrained(model)
        self.extract_feature = extract_feature

        self.max_tokens = max_tokens
        print("load %s"%data_path)

        data_pd = pd.read_csv(data_path, delimiter=",")
        if split=="train":
            data_pd = data_pd[data_pd["session"].isin(["Session%d"%i for i in list(range(1,cross_val))+list(range(cross_val+1,6))])]
        else:
            data_pd = data_pd[data_pd["session"]=="Session%d"%cross_val]
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
            self.label_gts[dialog].append(classes_name.index(label))
            # self.label_gts[dialog].append(label)
            # IEMOCAP/{session_id}/sestences/avi/{video}/{segment}.avi"
            fname = self.raw_path.format(session_id=session[-1], segment2=segment2)
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
    def batch_by_size(self, indices, max_tokens=None):
        batches = []
        batch_i = []
        acc_sum = 0
        for index_i in indices:
            if acc_sum+self.sizes[index_i] > max_tokens:
                batches.append(np.array(batch_i))
                acc_sum = self.sizes[index_i]
                batch_i = [index_i]
            else:
                acc_sum += self.sizes[index_i]
                batch_i.append(index_i)
        batches.append(np.array(batch_i))
        return batches
    def batch_sampler(self):
        return MyBatchSampler(self.batch_by_size(self.sizes.argsort(), max_tokens=self.max_tokens))
    def collate_fn(self, samples):
        with torch.no_grad():
            return {
                "pretrain_input": self.data_processor([feats.numpy() for sample in samples for feats in sample["source"]], padding=self.pad, sampling_rate=self.sample_rate, return_tensors="pt"),
                "dialog_lengths":  torch.tensor([sample["dialog_lengths"] for sample in samples]),
                "labels": torch.tensor([label for sample in samples for label in sample["label_gts"]]),
            }
            # return {"uttr_input": torch.cat(feats, axis=0), 
            #         "dialog_lengths":torch.LongTensor(dialog_lengths), 
            #         "labels":torch.LongTensor(labels)} # feats, dialog_length, label





class IEMOCAP5531CrossVal(Dataset):
    def __init__(self, 
                 split="train", 
                 cross_val=5, 
                 max_sentences = 120,
                 data_path = '/home/projects/12001458/chengqi/conv-emotion/AudioDialogueGCN/IEMOCAP_features/features/f/cross_val5/features/epoch=155-val_loss=3.81424-val_accuracy=0.71152_audio_linear.pkl', 
                 **kwards):
        feature_path = data_path.format(cross_val = cross_val)
        data = pickle.load(open(feature_path, 'rb'), encoding='latin1')
        print("load feature from %s"%feature_path)
        self.max_sentences = max_sentences
        self.features, self.labels =  data["features"], data["labels"]
        self.session_ids = data["session_ids"]
        cross_keys=[]        
        if split == "train":
            for i in list(range(1,cross_val))+list(range(cross_val+1,6)):
                cross_keys = cross_keys+self.session_ids["Session%d"%i]
        else:
            cross_keys = self.session_ids["Session%d"%cross_val]
        self.keys = data.get(split, cross_keys)
                
        self.sizes =  np.array([len(self.labels[id]) for id in self.keys], dtype=np.int64)
        self.len = len(self.keys)
        print(f"loaded {split} dataset: num={self.len}")

    def batch_sampler(self):
        assert self.sizes.max() <= self.max_sentences
        return MyBatchSampler(self.batch_by_size(self.sizes.argsort(), max_sentences=self.max_sentences))
    def batch_by_size(self, indices, max_sentences=None):
        batches = []
        batch_i = []
        acc_sum = 0
        for index_i in indices:
            if acc_sum+self.sizes[index_i] > max_sentences:
                batches.append(np.array(batch_i))
                acc_sum = self.sizes[index_i]
                batch_i = [index_i]
            else:
                acc_sum += self.sizes[index_i]
                batch_i.append(index_i)
        batches.append(np.array(batch_i))
        return batches
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
        return {
            "uttr_input": torch.cat(feats, axis=0), 
            "dialog_lengths":torch.LongTensor(dialog_lengths), 
            "labels":torch.LongTensor(labels)
            } # feats, dialog_length, label

class IEMOCAP5531CrossValWithFrame(IEMOCAP5531CrossVal):
    def __init__(self, 
                 split="train", 
                 cross_val=5, 
                 max_sentences = 120,
                 data_path = '',
                 **kwards):
        feature_path = data_path
        data = pickle.load(open(feature_path, 'rb'), encoding='latin1')
        print("load feature from %s"%feature_path)
        self.max_sentences = max_sentences
        self.features, self.labels =  data["features"], data["labels"]
        self.session_ids = data["session_ids"]
        cross_keys=[]
        if split == "train":
            for i in list(range(1,cross_val))+list(range(cross_val+1,6)):
                cross_keys = cross_keys+self.session_ids["Session%d"%i]
        else:
            cross_keys = self.session_ids["Session%d"%cross_val]  
        self.keys = data.get(split, cross_keys)
        self.sizes =  np.array([len(self.labels[id]) for id in self.keys], dtype=np.int64)
        self.len = len(self.keys)
        print(f"loaded {split} dataset: num={self.len}")

    def __getitem__(self, index):
        dialog_id = self.keys[index]
        # feats = torch.FloatTensor( np.stack([feat_i for feat_i in self.features[dialog_id]]))
        feats = [torch.from_numpy(feat_i).permute(1,0) for feat_i in self.features[dialog_id]]
        feats_length = [feat_i.shape[1] for feat_i in self.features[dialog_id]]
        dialog_length = len(self.labels[dialog_id])
        label = torch.LongTensor(self.labels[dialog_id])
        return feats, feats_length, dialog_length, label
    
    def collate_fn(self, data):
        feats = []
        frames_lengths = []
        dialog_lengths = []
        labels = []
        for data_i in data:
            feats.extend(data_i[0])
            frames_lengths.extend(data_i[1])
            dialog_lengths.append(data_i[2])
            labels.extend(data_i[3])
        return {"frames_inputs": pad_sequence(feats, True), 
                "frames_lengths":torch.LongTensor(frames_lengths), 
                "dialog_lengths":torch.LongTensor(dialog_lengths), 
                "labels":torch.LongTensor(labels)} # feats, dialog_length, label
  
class IEMOCAP5531CrossValWithFrameIntra(Dataset):
    def __init__(self, 
                 split="train", 
                 cross_val=5, 
                 data_path = 'data/myIemocap_4_5531/audio.pkl',
                 **kwards):
        feature_path = data_path
        data = pickle.load(open(feature_path, 'rb'), encoding='latin1')
        # print("load feature from %s"%feature_path)
        diaFeatures, dialabels =  data["features"], data["labels"]
        self.session_ids = data["session_ids"]
        cross_keys=[]
        if split == "train":
            for i in list(range(1,cross_val))+list(range(cross_val+1,6)):
                cross_keys = cross_keys+self.session_ids["Session%d"%i]
        else:
            cross_keys = self.session_ids["Session%d"%cross_val]  
            
        self.keys = data.get(split, cross_keys)

        self.features = []
        self.labels = []
        
        for dialog in self.keys:
            self.features.extend(diaFeatures[dialog])
            self.labels.extend(dialabels[dialog])
        print(f"loaded {split} dataset: num={len(self.features)}")
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return  torch.from_numpy(self.features[index]).permute(1,0), self.labels[index]
    def collate_fn(self, data):
        feats = []
        labels = []
        frames_lengths = []
        for data_i in data:
            feats.append(data_i[0])
            labels.append(data_i[1])
            frames_lengths.append(data_i[0].shape[0])
        return {"frames_inputs": pad_sequence(feats, True), 
                "frames_lengths":torch.LongTensor(frames_lengths), 
                "labels":torch.LongTensor(labels)} # feats, dialog_length, label

        
    
if __name__ == "__main__":
    # video_dataset = VideoIEMOCAPDataset(True)
    # audio_dataset = AudioIEMOCAPDataset(True)
    # data_0 = video_dataset[0]
    # dataset = IEMOCAPAudioDatasetsRaw("/home/users/ntu/n2107167/lnespnet/chengqi/ADGCNForEMC/datasets/iemocap_original/audio/IEMOCAP_Audio_4.csv",split="test")
    
    
    dataset = IEMOCAP5531CrossValWithFrameIntra("train")
    dataloader = DataLoader(dataset,
                            batch_size=120,
                            collate_fn=dataset.collate_fn,
                            num_workers=0,
                            pin_memory=False)
    for data in dataloader:
        data