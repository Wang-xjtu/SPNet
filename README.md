# SPNet

**Scale Propagation Network for Generalizable Depth Completion, 2024**

**Haotian Wang, Meng Yang, Xinhu Zheng, and Gang Hua**

![examples](https://github.com/user-attachments/assets/140d0a37-fb7f-4b91-ad6a-1cc1143e45ad)


## Requirments

Python=3.8

Pytorch=2.0 

## Training

The training code will be released after acceptance.

## Testing 

1. Download and save the pretrained model to checkpoints/

| Pretrained Model                                                                                    | Blocks    | Channels | Drop rate |
| --------------------------------------------------------------------------------------------------- |:-------:|:--------:|:-------:|
| [SPNet-Tiny](https://drive.google.com/file/d/1ivmCX-i9lej4uJhT0Yyk2Nq9ZmoQlsB9/view?usp=drive_link)    | [3,3,9,3]  | [96,192,384,768]    | 0.0  |
| [SPNet-Small](https://drive.google.com/file/d/1Ba-W3oX62lCjx5MvvGkn91LXP6SuCnV6/view?usp=drive_link)   | [3,3,27,3] | [96,192,384,768]    | 0.1  | 
| [SPNet-Base](https://drive.google.com/file/d/1B9uPRVPGm1F8F-isVDVzEdHgxXmp43hn/view?usp=drive_link)    | [3,3,27,3] | [128,256,512,1024]  | 0.1  | 
| [SPNet-Large](https://drive.google.com/file/d/11dujPviL4pKLEXytXK0mEmPBNQDqgEak/view?usp=drive_link)   | [3,3,27,3] | [192,384,768,1536]  | 0.2  | 

2. Download and unzip [test dataset](https://drive.google.com/file/d/10tME1cuV0PVxrFLauTlv5SdQbZLUfdGy/view?usp=drive_link)

3. Run test.py

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
