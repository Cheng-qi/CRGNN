# 对比论文实验结果统计

## ICASSP2023
### audio 
    IEMOCAP
    utterance: 5531
    conversations: 151
    l-o-se:leave-one-session-out  
    l-o-sp:leave-one-speaker-out
| Method | UA | WA | split | level |Features|url| remark |   
|---|---|---|---|---|---|---| ---|
|MSMSER(only audio)|64.9/63.4|63.2/62.9|l-o-se|uttrance|HuBERT & MPNet|[Exploring Complementary Features in Multi-Modal Speech Emotion Recognition ](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096709)
|DST|73.6|71.8|l-o-se|uttrance|WavLM|[DST: Deformable Speech Transformer for Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096966)
|SMW_CAT|74.25|73.8|l-o-sp|uttrance|Wav2vec2 & MFCC|[Multiple Acoustic Features Speech Emotion Recognition Using Cross-Attention Transformer](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095777)|不可参考|
|DCW + TsPA|72.17/74.26|72.08/73.18|l-o-se/l-o-sp|uttrance|Wav2vec2 & MFCC|[Speech Emotion Recognition Via Two-Stream Pooling Attention With Discriminative Channel Weighting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095588)||
|LabelAdaptiveMixup-SER|76.04|75.37|l-o-se|uttrance|HuBERT-Large+***Finetune***|[Learning Robust Self-Attention Features for Speech Emotion Recognition with Label-Adaptive Mixup](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095611)|结果挺严谨，且有源码，建议参考|
|DKDFMH|77.0|79.1|randomly train:80%, tret:20%|uttrance ***存疑***|logF-Bank|[Hierarchical Network with Decoupled Knowledge Distillation for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095045)|存疑，需要一起讨论|
|-|F1=70.1|-|l-o-se|uttrance|WavLM Large+***Finetune***|[Domain Adaptation without Catastrophic Forgetting on a Small-Scale Partially-Labeled Corpus for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096578)|不建议对比|
|-|73.86/70.01/75.60|-|l-o-se|uttrance|HuBERT large/Wav2vec 2.0/WavLM Large ***all Finetune***|[Speech-Based Emotion Recognition with Self-Supervised Models Using Attentive Channel-Wise Correlations and Label Smoothing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10094673)||
|TIM-Net|72.5|-|l-o-sp|uttrance|MFCC|[Temporal Modeling Matters: A Novel Temporal Emotional Modeling Approach for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096370)|有源码，给了特征，可信度较高，可参考|
|TFA+TFW+BCNN|79.07|81.57|l-o-se|uttrance|MFCC|[Speech Emotion Recognition Based on Low-Level Auto-Extracted Time-Frequency Features](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095260)|结果高的离谱|
|cFW-VCs|CCC = 63.6/71.9| |l-o-se|uttrance|LLDs/Wav2Vec2-large|[Role of Lexical Boundary Information in Chunk-Level Segmentation for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096861)|Chunk-level 估计不能用|
|P-TAPT|74.3|-|l-o-se|uttrance|LLDs/Wav2Vec2-finetune|[Exploring Wav2vec 2.0 Fine Tuning for Improved Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095036)||
|DWFormer|73.9|72.3|l-o-se|uttrance|WavLm-large|[DWFormer: Dynamic Window Transformer for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10094651)||
|ShiftCNN/ShiftLSTM/Shiftformer|74.5/74.7/74.8|-|l-o-se|uttrance|wav2vec2+***finetune***+Hubert+***finetune***|[Mingling or Misalignment? Temporal Shift for Speech Emotion Recognition with Pre-Trained Representations](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095193)|这篇文章对是否finetune分开进行了讨论，给了源码，参考意义较大|
|同上|72.8/70.6/72.7|71.9/69.8/72.1|同上|同上|wav2vec2|同上|
|-|-|-|-|-|-|[Designing and Evaluating Speech Emotion Recognition Systems: A Reality Check Case Study with IEMOCAP](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096808)|综述性文章，里面结果可以做参考|
|EMix-S|71.85|77.63|speaker-dependent+5fold|uttrance| log-Mel magnitude spectrogram|[EMIX: A Data Augmentation Method for Speech Emotion Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096789)|5fold交叉验证方法待讨论|






### Mutil-Modal
    IEMOCAP
    utterance: 7433
    conversations: 151

| Method | F1-Weighted|UA | WA | split | level |SSL|url | remark |   
|---|---|---|---|---|---|---| ---|---|
|MSRFG|71.60|-|-|leave last 20 conversations for test|dialog|Wav2vec2 & Roberta-Large|[Multi-Scale Receptive Field Graph Model for Emotion Recognition in Conversations](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10094596)|audio+text|内含多个结果可参考|



## 验证
1. /data/ADGCNForEMC/data/featuresFromPapers/IEMOCAP.npy 根据label判断顺序，基本性能, (librosa toolbox)
2. wavlm_large + dialog : >80%?  token?
3. 2022 icassp 总结, 语音，多模态   TAC, IEEE trans  

