# Learnable Graph Convolutional Network and Feature Fusion (LGCN-FF)

## Introduction
- This is an implement of LGCN-FF with PyTorch, which was run on a machine with AMD R9-5900HX CPU, RTX 3080 16G GPU and 32G RAM. It has been accepted by Information Fusion.

## Paper
Zhaoliang Chen, Lele Fu, Jie Yao, Wenzhong Guo, Claudia Plant and Shiping Wang. "Learnable Graph Convolutional Network and Feature Fusion for Multi-view Learning." Information Fusion (2023).


## Requirements
- torch: 1.11.0 + cu115
- numpy: 1.20.1
- scipy: 1.6.2
- scikit-learn: 1.1.2

## Running Examples
  - For ALOI: 
    ```
    python ./main.py --dataset-name ALOI --k 15 --epoch-num 180 
    ```
  - For BBCnews
    ```
    python ./main.py --dataset-name BBCnews --k 20 --epoch-num 300
    ```
  - For BBCsports
    ```
    python ./main.py --dataset-name BBCsports --k 14 --epoch-num 170
    ```
  - For Wikipedia
    ```
    python ./main.py --dataset-name Wikipedia --k 20 --epoch-num 150
    ```
  - For MSRC-v1
    ```
    python ./main.py --dataset-name MSRC-v1 --k 10 --epoch-num 130
    ```
  - For MNIST
    ```
    python ./main.py --dataset-name MNIST --k 12 --epoch-num 320
    ```

## Cite
```
@article{chen2023learnable,
  title = {Learnable Graph Convolutional Network and Feature Fusion for Multi-view Learning},
  author = {Chen, Zhaoliang and Fu, Lele and Yao, Jie and Guo, Wenzhong and Plant, Claudia and Wang, Shiping},
  journal = {Information Fusion},
  year = {2023},
  pages = {109-119},
  volume = {95},
}
```