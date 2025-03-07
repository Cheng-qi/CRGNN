# 对比论文实验结果统计

## ICASSP2023
### audio 
    IEMOCAP
    utterance: 5531
    conversations: 151
    l-o-se:leave-one-session-out  
    l-o-sp:leave-one-speaker-out
| Method | UA | WA | split | level |Features SSL | url| remark |   
|---|---|---|---|---|---|---| ---|
|MSMSER(only audio)|64.9/63.4|63.2/62.9|l-o-se|uttrance|HuBERT & MPNet|[Exploring Complementary Features in Multi-Modal Speech Emotion Recognition ](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096709)
|DST|73.6|71.8|l-o-se|uttrance|WavLM|[DST: Deformable Speech Transformer for Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096966)
|SMW_CAT|74.25|73.8|l-o-sp|uttrance|Wav2vec2 & MFCC|[Multiple Acoustic Features Speech Emotion Recognition Using Cross-Attention Transformer](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095777)|不可参考|
|DCW + TsPA|72.17/74.26|72.08/73.18|l-o-se/l-o-sp|uttrance|Wav2vec2 & MFCC|[Speech Emotion Recognition Via Two-Stream Pooling Attention With Discriminative Channel Weighting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095588)||
|LabelAdaptiveMixup-SER|76.04|75.37|l-o-se|uttrance|HuBERT-Large+***Finetune***|[Learning Robust Self-Attention Features for Speech Emotion Recognition with Label-Adaptive Mixup](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095611)|结果挺严谨，且有源码，建议参考|
|DKDFMH|77.0|79.1|randomly train:80%, tret:20%|uttrance ***存疑***|logF-Bank|[Hierarchical Network with Decoupled Knowledge Distillation for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095045)|存疑，需要一起讨论|
|-|F1=71|.-|l-o-se|uttrance|WavLM Large+***Finetune***|[Domain Adaptation without Catastrophic Forgetting on a Small-Scale Partially-Labeled Corpus for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096578)|不建议对比|
|-|73.86/701./75.60|-|l-o-se|uttrance|HuBERT large/Wav2vec 2.0/WavLM Large ***all Finetune***|[Speech-Based Emotion Recognition with Self-Supervised Models Using Attentive Channel-Wise Correlations and Label Smoothing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10094673)||
|TIM-Net|72.5|-|l-o-sp|uttrance|MFCC|[Temporal Modeling Matters: A Novel Temporal Emotional Modeling Approach for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096370)|有源码，给了特征，可信度较高，可参考|
|TFA+TFW+BCNN|79.07|81.57|l-o-se|uttrance|MFCC|[Speech Emotion Recognition Based on Low-Level Auto-Extracted Time-Frequency Features](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095260)|结果高的离谱|
|cFW-VCs|CCC = 63.6/71.9| |l-o-se|uttrance|LLDs/Wav2Vec2-large|[Role of Lexical Boundary Information in Chunk-Level Segmentation for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096861)|Chunk-level 估计不能用|
|P-TAPT|74.3|-|l-o-se|uttrance|LLDs/Wav2Vec2-finetune|[Exploring Wav2vec 2.0 Fine Tuning for Improved Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095036)||
|DWFormer|73.9|72.3|l-o-se|uttrance|WavLm-large|[DWFormer: Dynamic Window Transformer for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10094651)||
|ShiftCNN/ShiftLSTM/Shiftformer|74.5/74.7/74.8|-|l-o-se|uttrance|wav2vec2+***finetune***+Hubert+***finetune***|[Mingling or Misalignment? Temporal Shift for Speech Emotion Recognition with Pre-Trained Representations](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095193)|这篇文章对是否finetune分开进行了讨论，给了源码，参考意义较大|
|同上|72.8/76/.72.7|71.9/69.8/72.1|同上|同上|wav2vec2|同上|
|-|-|-|-|-|-|[Designing and Evaluating Speech Emotion Recognition Systems: A Reality Check Case Study with IEMOCAP](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096808)|综述性文章，里面结果可以做参考|
|EMix-S|71.85|77.63|speaker-dependent+5fold|uttrance| log-Mel magnitude spectrogram|[EMIX: A Data Augmentation Method for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096789)|5fold交叉验证方法待讨论|

## ICASSP 2022
| Method | UA | WA | split | level | SSL | url| remark |   
|---|---|---|---|---|---|---| ---|
| GLAM | 73.90 | 73.70 | 0.8train: 0.2test | utterance | MFCC | [Speech Emotion Recognition with Global-Aware Fusion on Multi-Scale Feature Representation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747517) |  improv, script 分开统计 |
| DAE+Linear-SVM | 52.09 | - | l-o-se | utterance | eGeMAPS | [Towards Transferable Speech Emotion Representation: On Loss Functions for Cross-Lingual Latent Representations](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746450) |   |
| CNN_SeqCap | 56.91 | 70.54 | l-o-se | utterance | spectrograms | [Neural Architecture Search for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746155) |   |
| LIGHT-SERNET | 70.76 | 70.23 | l-o-sp | utterance | MFCCs | [LIGHT-SERNET: A Lightweight Fully Convolutional Neural Network for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746679) | improv, script 分开统计 |
| ECAPA | 77.76 | 77.36 | l-o-se | utterance | HuBERT+W2V2 | [Speech Emotion Recognition Using Self-Supervised Features](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747870) |  |
| CKE-Net | 66.5 | - | 4810/1000/1523(7433) | conversation | HuBERT+W2V2 | [A Commonsense Knowledge Enhanced Network with Retrospective Loss for Emotion Recognition in Spoken Dialog](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746909) | IEMCAP7433 |
| TAP | 74.3 | - | l-o-se | utterance | HuBERT Large | [Speaker Normalization for Self-Supervised Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747460) |  |
| Co-Attention | 71.05 | 69.80 | l-o-se | utterance | MFCC+wav2vec2+spectrograms | [Speech Emotion Recognition with Co-Attention Based Multi-Level Acoustic Information](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747095) |  |
| Co-Attention | 72.70 | 71.64 | l-o-sp | utterance |  | [Speech Emotion Recognition with Co-Attention Based Multi-Level Acoustic Information](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747095) |  |
| averaged models | 75.2 | - | l-o-se | utterance | HuBERT Large finetuned | [Towards A Common Speech Analysis Engine](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747756) |  |

## other pub
| Method | dataset & split | UA | WA | level | SSL | url | source | remark |   
|---|---|---|---|---|---|---| ---| --- |
| 1D-MESA |  improvised IEMACAP (l-o-se) | 78.98 | 81.18 | uttrance | MFCCs | [Speech Emotion Recognition Based on Discriminative Features Extraction](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9859862) | ICME 2022 | |
| MLAnet |  improvised IEMACAP (l-o-se) | 80.05 | 82.46 | uttrance | MFCCs | [Speech Emotion Recognition via Multi-Level Attention Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9937073) | SPL 2022 | 双库 IEMOCAP & RAVDESS |
| DIFL_VGG |  EMO-DB (l-o-sp) | 88.49 | 89.72 | uttrance | Mel-spectrogram | [Domain Invariant Feature Learning for Speaker-Independent Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9836236) | TASLP 2022 | |
| m1 |  IEMOCAP(8:1:1) | 44.7 | - | uttrance | Mel spectrograms | [A Comparison Between Convolutional and Transformer Architectures for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9891882) | IJCNN 2022 | |
| m2 |  IEMOCAP(8:1:1) | 73.16 | - | uttrance | Wav2Vec2-large | - | IJCNN 2022 | |



## Mutil-Modal

### datasets
| name | uttrance | conversation | 
|---|---|---|
| IEMOCAP7433 | 7433 | 151 |
| IEMOCAP5531 | 5531 | 151 |


| Method | F1-Weighted|UA | WA | split | level | SSL | url | dataset | source | remark |   
|---|---|---|---|---|---|---| ---|---|---|---|
|MSRFG|71.60|-|-|leave last 20 conversations for test|dialog|a: Wav2vec2 + t: Roberta-Large|[Multi-Scale Receptive Field Graph Model for Emotion Recognition in Conversations](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10094596)|IEMACP7433 | ICASSP 2023 | 内含多个结果可参考|
|SMCN|-|77.6|75.6|l-o-se| conversation |a: & t: |[Multi-Modal Emotion Recognition with Self-Guided Modality Calibration](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747859) | IEMACP5531 | ICASSP 2022 | | 
|SMCN|62.3|64.9| - | conversation | - | - | MELD | | |  
|IMAN|64.5| - | 65.0 | conversation | - | [Interactive Multimodal Attention Network for Emotion Recognition in Conversation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9427095) | IEMOCAP7433 | SPL | |  






## result

### TIM-Net_SER MFCC features + TFCNN 
 /data/ADGCNForEMC/data/featuresFromPapers/IEMOCAP.npy 根据label判断顺序，基本性能, (librosa toolbox)
    使用[源码](https://github.com/Jiaxin-Ye/TIM-Net_SER)重新提取
**result(best)TFCNN**

| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 |53.06 | 55.04 |
| 1023 | session2 |55.43 | 55.10 |
| 1151 | session3 |54.56 | 53.91 |
| 1031 | session4 |53.06 | 55.04 |
| 1241 | session5 |52.62 | 54.73 |
| 5531 | mean | 53.75*(53.71) | 54.76(54.75) |
  
**:weighted*

### wavlm_large   

wavlm_large + CNNSelfAttention 5fold

| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 |72.63 | 73.01 |
| 1023 | session2 |78.89 | 847. |
| 1151 | session3 |74.72 | 74.44 |
| 1031 | session4 |78.66 | 77.89 |
| 1241 | session5 |73.01 | 73.95 |
| 5531 | mean | 75.43(75.58) | 75.86(75.95) |

wavlm_large + CNNSelfAttention + dialogGCN 5fold

| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 |75.48 | 77.31 |
| 1023 | session2 |82.21 | 81.82 |
| 1151 | session3 |77.50 | 76.89 |
| 1031 | session4 |75.85 | 73.31 |
| 1241 | session5 |79.94 | 79.13 |
| 5531 | mean | 78.21*(78.20)| 77.72(77.69) |
**:weighted*
wavlm_large + CNNSelfAttention + GCNII 5fold

| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 | 77.70 | 80.36 |
| 1023 | session2 | 84.07 | 85.04 |
| 1151 | session3 | 78.80 | 78.63 |
| 1031 | session4 | 78.95 | 78.05 |
| 1241 | session5 | 75.75 | 76.08 |
| 5531 | mean | 78.90*(79.05)| 79.47(79.63) |
**:weighted*

wavlm_large + CNNSelfAttention + ADGCN 5fold

| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 | 78.62 | 80.95 |
| 1023 | session2 | 83.28 | 84.47 |
| 1151 | session3 | 79.32 | 79.02 |
| 1031 | session4 | 79.44 | 76.85 |
| 1241 | session5 | 77.60 | 76.61 |
| 5531 | mean | 79.55*(79.65)| 79.46(79.58) |
**:weighted*


wavlm_large + CNNSelfAttention(f1) + CausalDiaModel &(MutilHeadAttention) ADGCN (Wavlm_CAL_ADGCN) flod5

| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 | 77.14 | 79.29 |
| 1023 | session2 | 82.50 | 82.64 |
| 1151 | session3 | 79.15 | 78.77 |
| 1031 | session4 | 80.50 | 79.56 |
| 1241 | session5 | 78.89 | 77.68 |
| 5531 | mean | (79.57)79.636 | (79.49)79.588 |
**:weighted*


wavlm_large + CNNSelfAttention(f1) + CausalDiaModel &(MutilHeadAttention) ADGCN flod5 no_valid

| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 | 78.25 | 80.85 |
| 1023 | session2 | 83.48 | 82.71 |
| 1151 | session3 | 81.58 | 81.41 |
| 1031 | session4 | 83.32 | 83.37 |
| 1241 | session5 | 79.53 | 78.70 |
| 5531 | mean | 81.142(81.232)| 81.298(81.408) |
**:weighted*




wavlm_large + CNNSelfAttention(f1) + CausalDiaModel + ADGCN 5fold (Wavlm_CAL_ADGCN) no_valid

| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 | 76.13 | 80.07 |
| 1023 | session2 | 84.56 | 85.70 |
| 1151 | session3 | 80.19 | 80.01 |
| 1031 | session4 | 79.44 | 76.48 |
| 1241 | session5 | 81.63 | 80.97 |
| 5531 | mean | 80.39(80.39) | 80.63(80.65) |
**:weighted*


3. 2022 icassp 总结, 语音，多模态   TAC, IEEE trans  





### plan
Deep graph For context application
1. 绪论(背景, 国内研究现状, 问题分析)
2. GNN相关方法概述(GNN, DeepGNN, 问题+解决方案) + context information fusion? & model?

3. Deep graph 理论 MAGCN
4. DiaMAGCN: MAGCN + Dialog  for 对话  audio + dialog: global, context,  
5. MAGCN + Dialog + mutilModal  for 多模态 vat + dialog: 




###  MAGCN + Dialog plan 
1. SPL 5页
2. **local context feature** + ADGCN
3. 实验：
    1. **对比结果 大表(DialogGCN, DialogRNN)**
    2. diff layers
    3. 消融实验: 
        1. local context feature with / without 
        2. single layer / ADGCN

4. 参考文献
    IEMOCAP 2023, 2022, 2021 
    SPL 
    DialogGCN 相关


## 紧急
1. **local context feature** 测试性能
2. MELD的实验
3. 找文献

4. MELD 再优化 
5. SPL 再找找文献
6. CONCAT [context, origin]

### 结果总结表
Wavlm_CAL 5fold
| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 | 73.09 | 76.58 |
| 1023 | session2 | 77.71 | 79.19 |
| 1151 | session3 | 76.37 | 76.53 |
| 1031 | session4 | 75.36 | 74.81 |
| 1241 | session5 | 78.08 | 77.47 |
| 5531 | mean | 76.17*(76.12)| 76.92(76.92) |
**:weighted*

Wavlm_DialigGCN 5fold
| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 | 77.14 | 77.43 |
| 1023 | session2 | 79.18 | 81.27 |
| 1151 | session3 | 75.41 | 75.54 |
| 1031 | session4 | 74.98 | 74.27 |
| 1241 | session5 | 73.57 | 76.14 |
| 5531 | mean | 75.95*(76.06)| 76.89(76.93) |
**:weighted*

Wavlm_ADGCN 5fold
| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 | 77.51 | 80.41 |
| 1023 | session2 | 83.09 | 83.88 |
| 1151 | session3 | 78.45 | 77.23 |
| 1031 | session4 | 77.50 | 74.37|
| 1241 | session5 | 74.46 | 72.94 |
| 5531 | mean | 78.05*(78.20)| 77.59(77.77) |
**:weighted*

Wavlm_CAL_ADGCN flod5
| num | session | WA | UA |  
| --- | --- | ---| --- |
| 1085 | session1 | 77.14 | 79.29 |
| 1023 | session2 | 82.50 | 82.64 |
| 1151 | session3 | 79.15 | 78.77 |
| 1031 | session4 | 80.50 | 79.56 |
| 1241 | session5 | 78.89 | 77.68 |
| 5531 | mean | 79.57(79.636) | 79.49(79.59) |
**:weighted*


| method               |IEMACAP|       |       |       |       |       |   | MELD  |       |
| ---                  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |---|  ---  |  ---  |
|                      | 
|                      |   N   |   A   |   S   |   H   |  WA   |  UA   |   |  WA   | F1-w  |
| Wavlm                | 71.12 | 80.54 | 75.46 | 76.11 | 75.43 | 75.81 |   |       |       |
| Wavlm_no_finetune    | 66.29 | 76.93 | 66.01 | 62.79 | 67.82 | 68.00 |   |       |       |
| DST                  | 71.70 | 79.96 | 76.20 | 75.90 | 75.46 | 75.94 |   |   -   | 48.80*|
| LabelAdaptiveMixup   | 68.11 | 77.59 | 74.40 | 77.72 | 74.29 | 74.46 |   |       |       |
| DWFormer             | 71.47 | 80.27 | 79.00 | 74.18 | 75.52 | 76.23 |   |       |       |
| Wavlm_DialogCRN      | 79.61 | 81.26 | 79.03 | 82.38 | 80.87 | 80.57 |   |       |       |
| Wavlm_DialogCRN_prue | 67.28 | 80.76 | 78.83 | 74.47 | 75.19 | 75.34 |   |       |       |
| Wavlm_DialogRNN      | 67.59 | 68.47 | 75.19 | 79.26 | 73.13 | 72.63 |   |       |       |
| Wavlm_DialogGCN      | 70.54 | 84.06 | 79.17 | 73.69 | 75.95 | 76.87 |   |       |       |
| Wavlm_CAL(our)       | 61.88 | 78.74 | 82.03 | 85.03 | 76.17 | 76.92 |   | 49.62 | **38.33** |
| Wavlm_ADGCN(our)     | 77.02 | 75.99 | 76.96 | 80.38 | 78.05 | 77.59 |   | 52.72 | 50.20 |
| Wavlm_CAL_ADGCN(our) | 76.36 | 78.13 | 82.14 | 81.34 | 79.57 | 79.49 |   | 53.91 | 50.18 |



no_finetune

| method               |IEMACAP|       |       |       |       |       |   | MELD  |       |
| ---                  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |---|  ---  |  ---  |
|                      | 
|                      |   N   |   A   |   S   |   H   |  WA   |  UA   |   |  WA   | F1-w  |
| Wavlm_CNNSA          | 66.29 | 76.93 | 66.01 | 62.79 | 67.82 | 68.00 |   |       |       |
<!-- | Wavlm_DialogCRN      | 67.28 | 80.76 | 78.83 | 74.47 | 75.19 | 75.34 |   |       |       | -->
| Wavlm_DWFormer       | 59.95 | 77.72 | 70.44 | 47.24 | 62.16 | 63.84 |   |       |       |






| method               |IEMACAP|       |       |       |       |       |   | MELD  |       |
| ---                  |  ---  |  ---  |  ---  |  ---  |  ---  |  ---  |---|  ---  |  ---  |
|                      | 
|                      |   N   |   A   |   S   |   H   |  WA   |  UA   |   |  WA   | F1-w  |
| Wavlm                | 66.29 | 76.93 | 66.01 | 62.79 | 67.82 | 68.00 |   |       |       |   
| DST                  | 71.70 | 79.96 | 76.20 | 75.90 | 75.46 | 75.94 |   |   -   | 48.80*|
| LabelAdaptiveMixup   | 68.11 | 77.59 | 74.40 | 77.72 | 74.29 | 74.46 |   |       |       |
| DWFormer             | 71.47 | 80.27 | 79.00 | 74.18 | 75.52 | 76.23 |   |       |       |
| DialogCRN            | 67.28 | 80.76 | 78.83 | 74.47 | 75.19 | 75.34 |   |       |       |
| DialogRNN            | 67.59 | 68.47 | 75.19 | 79.26 | 73.13 | 72.63 |   |       |       |
| DialogGCN            | 70.54 | 84.06 | 79.17 | 73.69 | 75.95 | 76.87 |   |       |       |
| Wavlm_CAL(our)       | 67.80 | 75.76 | 82.04 | 82.03 | 76.35 | 76.90 |   | 49.62 |       |
| Wavlm_ADGCN(10layer) | 77.02 | 75.99 | 76.96 | 80.38 | 78.05 | 77.59 |   | 52.72 | 50.20 |
| CAL_ADGCN_concat     | 72.59 | 82.23 | 84.02 | 74.38 | 77.33 | 78.31 |
| Wavlm_CAL_ADGCN(our) | 75.36 | 80.86 | 80.60 | 83.47 | 79.93 | 80.08 |   | 53.91 | 50.18 |
| CAL_ADGCN_2layer     | 77.06 | 79.19 | 84.07 | 79.32 | 79.53 | 79.91 |   | | |
| ADGCN_1layer         | 72.16 | 83.22 | 77.57 | 82.00 | 78.41 | 78.74 |   | | |
| ADGCN_3layer         | 76.17 | 78.1  | 75.47 | 80.70 | 78.00 | 77.61 |   | | |
| ADGCN_4layer         | 72.37 | 79.85 | 80.05 | 77.18 | 76.86 | 77.36 |   | | |
| ADGCN_5layer         | 75.01 | 80.92 | 76.83 | 77.48 | 77.40 | 77.56 |   | | |
| ADGCN_6layer         | 75.39 | 78.92 | 78.64 | 75.54 | 76.86 | 77.12 |   | | |
| ADGCN_7layer         | 71.86 | 80.11 | 78.22 | 81.09 | 77.71 | 77.82 |   | | |
| ADGCN_8layer         | 71.56 | 80.47 | 79.50 | 78.21 | 76.99 | 77.44 |   | | |
| ADGCN_9layer         | 76.68 | 78.67 | 76.39 | 76.55 | 77.17 | 77.07 |   | | |
| ADGCN_15layer        | 75.69 | 77.84 | 77.28 | 77.68 | 77.04 | 77.12 |   | | |
| ADGCN_20layer        | 77.06 | 81.52 | 77.29 | 74.36 | 76.99 | 77.56 |   | | |
| ADGCN_25layer        | 72.07 | 83.71 | 74.76 | 73.04 | 75.32 | 75.89 |   | | |
| ADGCN_30layer        | 75.92 | 72.40 | 80.94 | 79.83 | 76.91 | 77.27 |   | | |
| GCN_1layer_with_res  | 73.19 | 80.56 | 76.15 | 74.58 | 75.70 | 76.12 |   | | |
| GCN_2layer_with_res  | 70.79 | 83.60 | 76.19 | 75.93 | 76.10 | 76.63 |   | | |
| GCN_3layer_with_res  | 71.02 | 76.49 | 77.63 | 75.79 | 74.31 | 75.23 |   | | |
| GCN_4layer_with_res  | 68.02 | 79.83 | 78.99 | 76.51 | 75.25 | 75.84 |   | | |

两个表 一个图

## next 
1. 加IEMOCAP分类统计结果 
2. pretrain + new_model +source


## 
1. 补充一到两个期刊的对比实验


1 5 10 15 20 25 30
PINN


## 
1. 看看MELD的对话长度


##  shift result
### method1 
shift定义：一个dia中，前后两句不一样为一个shift  
(shift 后一句正确的数量) / shift数量
| method    | error_num | total_num | error_rate|
| ---       | ---       | ---       | --- |
|  CAL_w5   | 526 | 1449 | 0.3630 | | 826.0 | 1353 | 0.6105 |
|DialogRNN  | 633 | 1449 | 0.4369 | 
|DialogCRN  | 582 | 1449 | 0.4017 |
|DialogGCN  | 539 | 1449 | 0.3720 |





### method2
shift定义：一个dia中，**同一个speaker** 前后两句不一样为一个shift  
(shift 中后一句正确的数量) / shift数量
| method    | error_num  | total_num | error_rate|
| ---       | ---        | ---       | --- |
|  CAL_w5   | 306 | 793  | 0.3859 |
|DialogRNN  | 342 | 793  | 0.4313 |
|DialogCRN  | 385 | 793  | 0.4855 |
|DialogGCN  | 305 | 793  | 0.3846 |
            



## IEMOCAP
一共包含151段对话， 平均对话长度5531/151=37
