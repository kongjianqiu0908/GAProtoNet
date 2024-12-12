# GAProtoNet: A Multi-head Graph Attention-based Prototypical Network for Interpretable Binary Text Classification

## Introduction
This repository is the official implementation of "GAProtoNet: A Multi-head Graph Attention-based Prototypical Network for Interpretable Binary Text Classification". In this work, we introduce GAProtoNet, a novel white-box Multi-head Graph Attention-based Prototypical Network designed to explain the decisions of text classification models built with LM encoders. In our approach, the input vector and prototypes are regarded as nodes within a graph, and we utilize multi-head graph attention to selectively construct edges between the input node and prototype nodes to learn an interpretable prototypical representation. Experiments on multiple public datasets show our approach achieves superior results without sacrificing the accuracy of the original black-box LMs. We also compare with four alternative prototypical network variations and our approach achieves the best accuracy and F1 among all. Our case study and visualization of prototype clusters also demonstrate the efficiency in explaining the decisions of black-box models built with LMs.

#### Paper: [GAProtoNet: A Multi-head Graph Attention-based Prototypical Network for Interpretable Binary Text Classification](https://arxiv.org/abs/2409.13312)


## python 3.8
## create the environment
pip -r requirements.txt

## run the model
python src/train.py

