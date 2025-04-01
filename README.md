# T-PAMI2025: SPNet

**[Scale Propagation Network for Generalizable Depth Completion](https://ieeexplore.ieee.org/document/10786388)**

**Haotian Wang, Meng Yang, Xinhu Zheng, and Gang Hua**

**IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), March 2025**

## News

- Training code is released! `01/04/2025`

## Abstract

![examples](https://github.com/user-attachments/assets/140d0a37-fb7f-4b91-ad6a-1cc1143e45ad)

Depth completion, inferring dense depth maps from sparse measurements, is crucial for robust 3D perception. Although deep learning based methods have made tremendous progress in this problem, these models cannot generalize well across different scenes that are unobserved in training, posing a fundamental limitation that yet to be overcome. A careful analysis of existing deep neural network architectures for depth completion, which are largely borrowing from successful backbones for image analysis tasks, reveals that a key design bottleneck actually resides in the conventional normalization layers. These normalization layers are designed, on one hand, to make training more stable, on the other hand, to build more visual invariance across scene scales. However, in depth completion, the scale is actually what we want to robustly estimate in order to better generalize to unseen scenes. To mitigate, we propose a novel scale propagation normalization (SP-Norm) method to propagate scales from input to output, and simultaneously preserve the normalization operator for easy convergence. More specifically, we rescale the input using learned features of a single-layer perceptron from the normalized input, rather than directly normalizing the input as conventional normalization layers. We then develop a new network architecture based on SP-Norm and the ConvNeXt V2 backbone. We explore the composition of various basic blocks and architectures to achieve better performance and faster inference for generalizable depth completion. Extensive experiments are conducted on six unseen datasets with various types of sparse depth maps, i.e., randomly sampled 0.1\%/1\%/10\% valid pixels, 4/8/16/32/64-line LiDAR points, and holes from Structured-Light. Our model consistently achieves superior performance with faster speed and lower memory when compared to state-of-the-art methods.

## Requirments

Python=3.8

Pytorch=2.3

## Train

#### Prepare your data

1. save your rgbd datasets in `./RGBD_Datasets`

```
└── RGBD_Datasets
 ├── Dataset1
 │   ├── rgb
 │   │   ├── file1.png
 │   │   ├── file2.png
 │   │   └── ...
 │   └── depth
 │       ├── file1.png
 │       ├── file2.png
 │       └── ...
 └── Dataset2
     ├── rgb
     │   ├── file1.png
     │   ├── file2.png
     │   └── ...
     └── depth
         ├── file1.png
         ├── file2.png
         └── ...    
```

**Notably:** `depth` should be stored in 16-bit data. Specifically, depth maps are normalized by `depth/max_depth*65535`, where `max_depth` is `20`(m) for indoor dataset and `100`(m) for outdoor dataset. We release the [UnrealCV](https://drive.google.com/file/d/1svV_j8IwjH1fcF4iDtAAh4MRw0Ig00X-/view?usp=drive_link) dataset as one example.

2. save your hole datasets in `./Hole_Datasets`

```
└── Hole_Datasets
 ├── Dataset1
 │   ├── file1.png
 │   ├── file2.png
 │   └── ...
 └── Dataset2
     ├── file1.png
     ├── file2.png
     └── ...
```

**Notably:** hole maps should be stored in Uint8 format. Specifically, `pixels without holes = 255` and `pixels within holes = 0`. We release the [hole collected from HRWSI](https://drive.google.com/file/d/1iKJEWgd36ebEVbG-01_gDipYuCCs7ZQZ/view?usp=drive_link) dataset as one example.

#### Start your training

1. Run `train.py`

```
# model_type: ["Tiny", "Small", "Base", "Large"]
python train.py --model_type="Large"
```

2. The trained model is saved in `./checkpoints/models`

## Test

1. Download and save the `pretrained model` to `./checkpoints/models`

| Pretrained Model                                                                                    | Blocks    | Channels | Drop rate |
| --------------------------------------------------------------------------------------------------- |:-------:|:--------:|:-------:|
| [SPNet-Tiny](https://drive.google.com/file/d/1ivmCX-i9lej4uJhT0Yyk2Nq9ZmoQlsB9/view?usp=drive_link)    | [3,3,9,3]  | [96,192,384,768]    | 0.0  |
| [SPNet-Small](https://drive.google.com/file/d/1Ba-W3oX62lCjx5MvvGkn91LXP6SuCnV6/view?usp=drive_link)   | [3,3,27,3] | [96,192,384,768]    | 0.1  | 
| [SPNet-Base](https://drive.google.com/file/d/1B9uPRVPGm1F8F-isVDVzEdHgxXmp43hn/view?usp=drive_link)    | [3,3,27,3] | [128,256,512,1024]  | 0.1  | 
| [SPNet-Large](https://drive.google.com/file/d/11dujPviL4pKLEXytXK0mEmPBNQDqgEak/view?usp=drive_link)   | [3,3,27,3] | [192,384,768,1536]  | 0.2  | 

2. Download and unzip [`test dataset`](https://drive.google.com/file/d/10tME1cuV0PVxrFLauTlv5SdQbZLUfdGy/view?usp=drive_link) to `./Test_Datasets`

3. Run `test.py`

```
# model_type: ["Tiny", "Small", "Base", "Large"]
python test.py --model_type="Large"
```

**Notably:** `gt` in test data are also stored in 16-bit data. Specifically, depth maps are normalized by `gt/max_depth*65535`, where `max_depth` is `20`(m) for indoor dataset and `100`(m) for outdoor dataset.

**This repository adopts a similar framework to [G2-MonoDepth](https://github.com/Wang-xjtu/G2-MonoDepth).**

## Citation

```
@ARTICLE{10786388,
  author={Wang, Haotian and Yang, Meng and Zheng, Xinhu and Hua, Gang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Scale Propagation Network for Generalizable Depth Completion}, 
  year={2025},
  volume={47},
  number={3},
  pages={1908-1922},
  doi={10.1109/TPAMI.2024.3513440}}
```
