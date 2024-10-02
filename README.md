# SPNet

**Scale Propagation Network for Generalizable Depth Completion, 2024**

**Haotian Wang, Meng Yang, Xinhu Zheng, and Gang Hua**

## Abstract

![examples](https://github.com/user-attachments/assets/140d0a37-fb7f-4b91-ad6a-1cc1143e45ad)

Depth completion, inferring dense depth maps from sparse measurements, is crucial for robust 3D perception. Although deep learning based methods have made tremendous progress in this problem, these models cannot generalize well across different scenes that are unobserved in training, posing a fundamental limitation that yet to be overcome. A careful analysis of existing deep neural network architectures for depth completion, which are largely borrowing from successful backbones for image analysis tasks, reveals that a key design bottleneck actually resides in the conventional normalization layers. These normalization layers are designed, on one hand, to make training more stable, on the other hand, to build more visual invariance across scene scales. However, in depth completion, the scale is actually what we want to robustly estimate in order to better generalize to unseen scenes. To mitigate, we propose a novel scale propagation normalization (SP-Norm) method to propagate scales from input to output, and simultaneously preserve the normalization operator for easy convergence. More specifically, we rescale the input using learned features of a single-layer perceptron from the normalized input, rather than directly normalizing the input as conventional normalization layers. We then develop a new network architecture based on SP-Norm and the ConvNeXt V2 backbone. We explore the composition of various basic blocks and architectures to achieve better performance and faster inference for generalizable depth completion. Extensive experiments are conducted on six unseen datasets with various types of sparse depth maps, i.e., randomly sampled 0.1%/1%/10% valid pixels, 4/8/16/32/64-line LiDAR points, and holes from Structured-Light. Our model consistently achieves superior performance with faster or comparable speed when compared to existing methods.


## Requirments

Python=3.8

Pytorch=2.3 

## Training

Coming soon.

## Testing 

1. Download and save the `pretrained model` to `./checkpoints`

| Pretrained Model                                                                                    | Blocks    | Channels | Drop rate |
| --------------------------------------------------------------------------------------------------- |:-------:|:--------:|:-------:|
| [SPNet-Tiny](https://drive.google.com/file/d/1ivmCX-i9lej4uJhT0Yyk2Nq9ZmoQlsB9/view?usp=drive_link)    | [3,3,9,3]  | [96,192,384,768]    | 0.0  |
| [SPNet-Small](https://drive.google.com/file/d/1Ba-W3oX62lCjx5MvvGkn91LXP6SuCnV6/view?usp=drive_link)   | [3,3,27,3] | [96,192,384,768]    | 0.1  | 
| [SPNet-Base](https://drive.google.com/file/d/1B9uPRVPGm1F8F-isVDVzEdHgxXmp43hn/view?usp=drive_link)    | [3,3,27,3] | [128,256,512,1024]  | 0.1  | 
| [SPNet-Large](https://drive.google.com/file/d/11dujPviL4pKLEXytXK0mEmPBNQDqgEak/view?usp=drive_link)   | [3,3,27,3] | [192,384,768,1536]  | 0.2  | 

2. Download and unzip [`test dataset`](https://drive.google.com/file/d/10tME1cuV0PVxrFLauTlv5SdQbZLUfdGy/view?usp=drive_link)

3. Run `test.py`

```python
# SPNet-Tiny
python test.py --dims=[3,3,9,3] --depths=[96,192,384,768] --dp_rate=0.0 --model_dir='checkpoints/Tiny.pth'
# SPNet-Small
python test.py --dims=[3,3,27,3] --depths=[96,192,384,768] --dp_rate=0.1 --model_dir='checkpoints/Small.pth'
# SPNet-Base
python test.py --dims=[3,3,27,3] --depths=[128,256,512,1024] --dp_rate=0.1 --model_dir='checkpoints/Base.pth'
# SPNet-Large
python test.py --dims=[3,3,27,3] --depths=[192,384,768,1536] --dp_rate=0.2 --model_dir='checkpoints/Large.pth'
```
