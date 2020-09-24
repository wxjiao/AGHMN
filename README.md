# AGHMN: Real-Time Emotion Recognition via Attention GatedHierarchical Memory Network

Implementation of the paper [Real-Time Emotion Recognition via Attention GatedHierarchical Memory Network](https://aaai.org/ojs/index.php/AAAI/article/view/6309) in AAAI-2020.

(The scripts uploaded here need further clean, please wait.)

## Brief Introduction

Real-time emotion recognition (RTER) in conversations is significant for developing emotionally intelligent chatting machines. Without the future context in RTER, it becomes critical to build the memory bank carefully for capturing historical context and summarize the memories appropriately to retrieve relevant information.
We propose an Attention Gated Hierarchical Memory Network (AGHMN) to address the problems of prior work: (1) Commonly used convolutional neural networks (CNNs) for utterance feature extraction are less compatible in the memory modules; (2) Unidirectional gated recurrent units (GRUs) only allow each historical utterance to have context before it, preventing information propagation in the opposite direction; (3) The Soft Attention for summarizing loses the positional and ordering information of memories, regardless of how the memory bank is built.
Particularly, we propose a Hierarchical Memory Network (HMN) with a bidirectional GRU (BiGRU) as the utterance reader and a BiGRU fusion layer for the interaction between historical utterances. For memory summarizing, we propose an Attention GRU (AGRU) where we utilize the attention weights to update the internal state of GRU. We further promote the AGRU to a bidirectional variant (BiAGRU) to balance the contextual information from recent memories and that from distant memories. We conduct experiments on two emotion conversation datasets with extensive analysis, demonstrating the efficacy of our AGHMN models.

<div align="center">
    <img src="/image/AGHMN.png" width="80%" title="Framework of AGHMN."</img>
    <p class="image-caption">Figure 1: The framework of AGHMN.</p>
</div>


## Code Base

### Dataset

Please find the datasets:
- [IEMOCAP](https://sail.usc.edu/iemocap/): IEMOCAP contains approximately 12 hours of audiovisual data, including video, speech, motion capture of face, text transcriptions.
- [MELD](https://github.com/SenticNet/MELD): A multimodal multi-party dataset for emotion recognition in conversation. 


### Prerequisites
- Python v3.6
- Pytorch v0.4.0-v0.4.1
- Pickle


## Citation
Please kindly cite our paper:
```ruby
@inproceedings{jiao2020real,
  author    = {Wenxiang Jiao, Michael R. Lyu and Irwin King},
  title     = {Real-Time Emotion Recognition via Attention Gated Hierarchical Memory Network},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2020, The Thirty-Second Innovative Applications of Artificial Intelligence
               Conference, {IAAI} 2020, The Tenth {AAAI} Symposium on Educational
               Advances in Artificial Intelligence, {EAAI} 2020, New York, NY, USA,
               February 7-12, 2020},
  pages     = {8002--8009},
  year      = {2020}
}
```
